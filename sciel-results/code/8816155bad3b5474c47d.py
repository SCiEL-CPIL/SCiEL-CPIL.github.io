"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
import time

# EVOLVE-BLOCK-START

def _is_valid_subset(subset, n=255):
    """Check that subset is a valid non-coprime set containing n."""
    if n not in subset:
        return False
    if not subset or min(subset) < 1 or max(subset) > n:
        return False
    s = list(subset)
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if gcd(s[i], s[j]) == 1:
                return False
    return True

def _bron_kerbosch_max_independent_set(conflict_graph, n_nodes):
    """
    Find maximum independent set in conflict_graph using Bron-Kerbosch with pivot.
    conflict_graph: dict mapping node -> set of conflicting nodes
    Returns the largest independent set found.
    """
    best_set = set()
    
    # Use a bitmask-based approach for efficiency
    # Represent sets as Python sets for clarity
    
    def bk(R, P, X):
        nonlocal best_set
        if len(R) > len(best_set):
            best_set = R.copy()
        
        if not P and not X:
            return
        
        # Pivot selection: choose pivot from P ∪ X with maximum degree in P
        union_px = P | X
        if not union_px:
            return
        pivot = max(union_px, key=lambda v: len(P & conflict_graph.get(v, set())))
        
        # Candidates: P \\ N(pivot)
        candidates = P - conflict_graph.get(pivot, set())
        
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & conflict_graph.get(v, set())
            X_new = X & conflict_graph.get(v, set())
            # Note: in independent set context, neighbors in conflict graph are those we CANNOT take together
            # So P_new = P ∩ N(v) where N(v) is the set of nodes NOT conflicting with v
            # Wait, I need to be careful with the formulation.
            
            # Let me reformulate: We're finding maximum independent set.
            # In Bron-Kerbosch for independent set:
            # - P = candidates that can still be added
            # - X = candidates already excluded (for completeness)
            # - When we pick v, new P = P ∩ N_independent(v) where N_independent(v) = {u : u not in conflict(v)}
            # - new X = X ∩ N_independent(v)
            pass

    # Let me use a cleaner formulation
    # For maximum independent set, use Bron-Kerbosch on the complement graph (i.e., find max clique in complement)
    # Or equivalently, find max independent set directly.
    
    # Simpler approach: use a greedy + local search heuristic given time constraints
    
    return best_set

def _greedy_with_local_search(conflict_graph, n_nodes, deadline):
    """
    Greedy construction + local search to maximize independent set size.
    """
    # Build adjacency for non-conflict (i.e., pairs that CAN coexist)
    non_conflict = {}
    for i in range(n_nodes):
        non_conflict[i] = set()
        for j in range(i+1, n_nodes):
            if j not in conflict_graph.get(i, set()):
                non_conflict[i].add(j)
                non_conflict[j].add(i)
    
    # Greedy: repeatedly pick node with fewest conflicts among remaining
    remaining = set(range(n_nodes))
    selected = set()
    
    while remaining:
        # Pick node with minimum conflict count in remaining
        best_node = min(remaining, key=lambda v: len(conflict_graph.get(v, set()) & remaining))
        selected.add(best_node)
        remaining.discard(best_node)
        # Remove all nodes conflicting with best_node
        for v in conflict_graph.get(best_node, set()):
            remaining.discard(v)
    
    current_size = len(selected)
    best_set = set(selected)
    
    # Local search: try to swap out one node for one that allows more additions
    elapsed = time.time() - deadline[0]
    time_limit = 80  # seconds
    
    iteration = 0
    while elapsed < time_limit:
        improved = False
        # Try removing each selected node and re-adding optimally
        for v in list(best_set):
            trial = best_set - {v}
            # Try to add back a different node
            for u in range(n_nodes):
                if u in trial:
                    continue
                # Check if u can be added
                if all(u not in conflict_graph.get(w, set()) for w in trial):
                    new_set = trial | {u}
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
        elapsed = time.time() - deadline[0]
    
    return best_set

def construct():
    deadline = [time.time()]
    n = 255
    
    # Build the set of candidate numbers: those that share a common factor with 255
    # 255 = 3 * 5 * 17
    # Numbers in [1,255] that share a factor with 255: divisible by 3, 5, or 17
    candidates = []
    for i in range(1, n+1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Map candidate values to indices
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    n_nodes = len(candidates)
    
    # Build conflict graph: two numbers conflict if gcd == 1
    conflict_graph = {i: set() for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if gcd(candidates[i], candidates[j]) == 1:
                conflict_graph[i].add(j)
                conflict_graph[j].add(i)
    
    # Baseline: multiples of 3
    mult_of_3 = [i for i in range(3, n+1, 3)]
    baseline_idx = set(val_to_idx[v] for v in mult_of_3 if v in val_to_idx)
    baseline_size = len(baseline_idx)
    
    # Verify baseline is valid
    baseline_vals = [candidates[i] for i in baseline_idx]
    valid_baseline = _is_valid_subset(baseline_vals, n)
    
    best_set = baseline_idx.copy()
    best_size = baseline_size
    
    # Try to improve with greedy + local search
    if n_nodes < 200:  # Should be manageable
        result_idx = _greedy_with_local_search(conflict_graph, n_nodes, deadline)
        if len(result_idx) > best_size:
            # Validate
            result_vals = [candidates[i] for i in result_idx]
            if _is_valid_subset(result_vals, n):
                best_set = result_idx
                best_size = len(result_idx)
    
    # Final check: ensure 255 is in the set
    if 255 not in [candidates[i] for i in best_set]:
        # Fallback to multiples of 3
        best_set = baseline_idx.copy()
        best_size = baseline_size
    
    result_vals = sorted(candidates[i] for i in best_set)
    
    return result_vals

# EVOLVE-BLOCK-END
