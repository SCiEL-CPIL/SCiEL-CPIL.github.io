import time
import random
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    node_set = set(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    n = len(valid_nodes)
    
    # Build conflict adjacency: conflicts[i] = set of indices j where valid_nodes[i]+valid_nodes[j] is a square
    conflicts = [set() for _ in range(n)]
    for i_idx in range(n):
        v = valid_nodes[i_idx]
        for j_idx in range(i_idx + 1, n):
            w = valid_nodes[j_idx]
            if v + w in squares:
                conflicts[i_idx].add(j_idx)
                conflicts[j_idx].add(i_idx)
    
    # Build bitmask representation for fast conflict checking
    conflict_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
    
    def is_valid_set(selected):
        """Verify selected indices form an independent set."""
        sel_set = set(selected)
        for idx in sel_set:
            for nb in conflicts[idx]:
                if nb in sel_set:
                    return False
        return True
    
    def greedy_mis(order):
        """Greedy: process nodes in given order, add if no conflict with already selected."""
        selected = set()
        for idx in order:
            if all(nb not in selected for nb in conflicts[idx]):
                selected.add(idx)
        return selected
    
    def local_search(selected, max_iters=5000):
        """Try to improve by removing one and adding multiple, or swapping."""
        selected = set(selected)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and (time.time() - start_time) < deadline - 1:
            improved = False
            iter_count += 1
            # Try adding a single unselected node
            for idx in range(n):
                if idx not in selected:
                    if all(nb not in selected for nb in conflicts[idx]):
                        selected.add(idx)
                        improved = True
                        break
            if improved:
                continue
            # Try removing one node and adding multiple (efficient version)
            best_gain = 0
            best_new_sel = None
            for rem in list(selected):
                if (time.time() - start_time) > deadline - 1:
                    break
                new_sel = set(selected)
                new_sel.discard(rem)
                # Try to greedily add back as many as possible
                for idx in range(n):
                    if idx not in new_sel and idx != rem:
                        if all(nb not in new_sel for nb in conflicts[idx]):
                            new_sel.add(idx)
                gain = len(new_sel) - len(selected)
                if gain > best_gain:
                    best_gain = gain
                    best_new_sel = new_sel
            if best_gain > 0 and best_new_sel is not None:
                selected = best_new_sel
                improved = True
        return selected
    
    def ilp_mis():
        """Solve maximum independent set via ILP (LP relaxation + rounding + repair)."""
        # First get a good lower bound via greedy
        degrees = [len(conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        lb = greedy_mis(order)
        lb_size = len(lb)
        
        # Use LP relaxation with scipy
        # Maximize sum(x_i) subject to x_i + x_j <= 1 for each edge (i,j)
        c = -np.ones(n)  # minimize -sum(x_i)
        
        # Build constraint matrix
        edge_list = []
        for i in range(n):
            for j in conflicts[i]:
                if i < j:
                    edge_list.append((i, j))
        
        if not edge_list:
            return set(range(n))
        
        m = len(edge_list)
        A_ub = np.zeros((m, n))
        b_ub = np.ones(m)
        for k, (i, j) in enumerate(edge_list):
            A_ub[k, i] = 1
            A_ub[k, j] = 1
        
        bounds = [(0, 1)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Round the LP solution: take nodes with x_i > 0.5, then repair
                x = res.x
                selected = set()
                # Sort by LP value descending for better rounding
                order_by_lp = sorted(range(n), key=lambda i: -x[i])
                for i in order_by_lp:
                    if x[i] > 0.3:  # Lower threshold, repair will fix conflicts
                        if all(nb not in selected for nb in conflicts[i]):
                            selected.add(i)
                
                # If rounding gave poor result, fall back to greedy
                if len(selected) < lb_size:
                    return lb
                return selected
            else:
                return lb
        except Exception:
            return lb
    
    best = set()
    best_size = 0
    
    # Multiple greedy initializations with different orderings
    random.seed(42)
    
    # Order 1: natural order
    order = list(range(n))
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Order 2: reverse
    order = list(range(n - 1, -1, -1))
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Order 3: sorted by degree (fewest conflicts first)
    degrees = [len(conflicts[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # Order 4: sorted by degree (most conflicts first)
    order = sorted(range(n), key=lambda x: -degrees[x])
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)
    
    # ILP-based initialization
    if (time.time() - start_time) < deadline - 5:
        try:
            ilp_sel = ilp_mis()
            ilp_sel = local_search(ilp_sel, max_iters=10000)
            if len(ilp_sel) > best_size:
                best = ilp_sel
                best_size = len(ilp_sel)
        except Exception:
            pass
    
    # Random orderings
    for trial in range(50):
        if (time.time() - start_time) > deadline - 2:
            break
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_mis(order)
        sel = local_search(sel)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # More aggressive local search on best
    if (time.time() - start_time) < deadline - 3:
        best = local_search(best, max_iters=20000)
        best_size = len(best)
    
    # Additional random restarts with local search if time permits
    for trial in range(30):
        if (time.time() - start_time) > deadline - 2:
            break
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_mis(order)
        sel = local_search(sel, max_iters=5000)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)
    
    # Final verification
    if not is_valid_set(best):
        # Fallback: rebuild
        order = list(range(n))
        best = greedy_mis(order)
    
    # One more local search pass on the final result
    if (time.time() - start_time) < deadline - 1:
        best = local_search(best, max_iters=5000)
    
    # Convert back to values
    result = sorted(valid_nodes[i] for i in best)
    
    if not result:
        # Absolute fallback
        result = [1]
    
    return result
