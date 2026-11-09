"""Optimized disjoint residue construction: maximize sum 1/m with pairwise disjointness.
Uses exact branch-and-bound (maximum weight independent set) with bitset pruning."""
import time
import math
import random
from fractions import Fraction

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _build_conflict_graph():
    """Build node list and conflict adjacency (bitset per node) for all (m, r) pairs.
    Node id = (m-20)*200 + r. Node exists only if 20<=m<=200 and 0<=r<m.
    Edge between two nodes iff they are NOT disjoint (i.e., conflict).
    Returns (nodes, weights, adj) where nodes is list of (m, r), weights list of Fraction,
    adj is list of Python ints (bitsets).
    """
    nodes = []
    weights = []
    id_map = {}
    for m in range(20, 201):
        for r in range(m):
            idx = len(nodes)
            nodes.append((m, r))
            weights.append(Fraction(1, m))
            id_map[(m, r)] = idx
    n = len(nodes)
    adj = [0] * n
    for i in range(n):
        m1, r1 = nodes[i]
        for j in range(i + 1, n):
            m2, r2 = nodes[j]
            g = math.gcd(m1, m2)
            if (r1 % g) == (r2 % g):
                # conflict: not disjoint
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return nodes, weights, adj, id_map

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _score_fast(classes):
    """Fast float score for comparisons during search."""
    return sum(1.0 / m for m, r in classes)

def _bfs_bound(adj, candidates):
    """Upper bound via greedy coloring of the candidate subgraph.
    candidates is a Python int bitset. Returns (upper_bound_int, colors) where
    upper_bound is sum of weights of color classes (using max weight per color as bound),
    and colors is a dict node->color for ordering.
    We use a simple greedy: repeatedly pick the smallest uncolored node, color it with the
    smallest available color, and the bound is sum over colors of the max weight in that color.
    For MWIS bound we use: sum over colors of (max weight in that color class) as an upper
    bound on the weight of an independent set (since at most one node per color class).
    """
    if candidates == 0:
        return 0, {}
    uncolored = candidates
    colors = {}
    color_classes = []  # list of lists of (node, weight)
    while uncolored:
        # pick lowest set bit
        lsb = uncolored & (-uncolored)
        v = lsb.bit_length() - 1
        uncolored ^= lsb
        # find smallest color class that doesn't contain a neighbor of v
        # Actually for coloring bound: color v with smallest color c such that no node in color c is adjacent to v
        # But simpler: assign color = number of already-assigned colors that conflict... 
        # Standard greedy coloring: assign smallest color not used by neighbors.
        used_colors = set()
        # check neighbors of v that are already colored
        nbrs = adj[v] & candidates
        # iterate over colored neighbors
        tmp = nbrs
        while tmp:
            lsb2 = tmp & (-tmp)
            u = lsb2.bit_length() - 1
            tmp ^= lsb2
            if u in colors:
                used_colors.add(colors[u])
        c = 0
        while c in used_colors:
            c += 1
        colors[v] = c
        if c >= len(color_classes):
            color_classes.append([])
        color_classes[c].append(v)
    # bound: for each color class, at most one node can be in independent set
    # upper bound = sum over color classes of max weight in that class
    bound = 0
    for cc in color_classes:
        max_w = 0
        for v in cc:
            w = weights_global[v]
            if w > max_w:
                max_w = w
        bound += max_w
    return bound, colors

def _mwis_exact(adj, weights, deadline, init_solution=None):
    """Branch and bound for maximum weight independent set.
    adj: list of bitsets, weights: list of Fraction.
    Returns (best_weight, best_set) where best_set is a bitset.
    """
    global weights_global
    weights_global = weights
    n = len(weights)
    best_w = Fraction(0, 1)
    best_set = 0
    
    if init_solution is not None:
        best_w = Fraction(0, 1)
        for v in init_solution:
            best_w += weights[v]
        best_set = 0
        for v in init_solution:
            best_set |= (1 << v)
    
    # Use recursive branch-and-bound with bitsets
    # State: (candidates_bitset, current_weight, current_set)
    # Upper bound: greedy coloring of candidates
    
    import sys
    sys.setrecursionlimit(100000)
    
    def bound_fn(cands):
        if cands == 0:
            return Fraction(0, 1)
        # Quick bound: sum of all weights in cands (loose but fast)
        # Better: greedy coloring
        uncolored = cands
        color_classes = []
        while uncolored:
            lsb = uncolored & (-uncolored)
            v = lsb.bit_length() - 1
            uncolored ^= lsb
            # find color
            used = set()
            nbrs = adj[v] & cands
            tmp = nbrs
            while tmp:
                lsb2 = tmp & (-tmp)
                u = lsb2.bit_length() - 1
                tmp ^= lsb2
                if u in color_map:
                    used.add(color_map[u])
            c = 0
            while c in used:
                c += 1
            color_map[v] = c
            if c >= len(color_classes):
                color_classes.append(0)
            # track max weight in this color
            wv = weights[v]
            if wv > color_classes[c]:
                color_classes[c] = wv
        color_map.clear()
        total = Fraction(0, 1)
        for cw in color_classes:
            total += cw
        return total
    
    color_map = {}
    
    def branch(cands, cur_w, cur_set, depth):
        nonlocal best_w, best_set
        if time.time() > deadline:
            return
        if cands == 0:
            if cur_w > best_w:
                best_w = cur_w
                best_set = cur_set
            return
        # bound
        ub = cur_w + bound_fn(cands)
        if ub <= best_w:
            return
        # pick a vertex: highest weight in cands
        # find max weight node
        tmp = cands
        best_v = -1
        best_wv = Fraction(0, 1)
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            if weights[v] > best_wv:
                best_wv = weights[v]
                best_v = v
        if best_v < 0:
            return
        # Branch 1: include best_v
        new_set = cur_set | (1 << best_v)
        new_cands = cands & ~(1 << best_v)
        # remove neighbors of best_v from candidates
        new_cands &= ~adj[best_v]
        branch(new_cands, cur_w + weights[best_v], new_set, depth + 1)
        # Branch 2: exclude best_v
        new_cands2 = cands & ~(1 << best_v)
        branch(new_cands2, cur_w, cur_set, depth + 1)
    
    all_cands = (1 << n) - 1
    branch(all_cands, Fraction(0, 1), 0, 0)
    
    return best_w, best_set

def _find_valid_residues(m, selected):
    """Find all valid residues for modulus m given already selected classes."""
    valid = []
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            g = math.gcd(m, m2)
            if (r % g) == (r2 % g):
                ok = False
                break
        if ok:
            valid.append(r)
    return valid

def _local_search(classes, deadline):
    """Strong local search: add, swap (1-for-1), swap (1-for-2), residue reassignment."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_moduli = set(m for m, r in classes)
        base_score = _score_fast(classes)
        
        # 1) Try to add a new modulus (prefer smaller m)
        for m in range(20, 201):
            if time.time() > deadline:
                return classes
            if m in current_moduli:
                continue
            valid_rs = _find_valid_residues(m, classes)
            if valid_rs:
                new_classes = classes + [[m, valid_rs[0]]]
                if _is_valid(new_classes):
                    sc = _score_fast(new_classes)
                    if sc > base_score + 1e-12:
                        classes = new_classes
                        current_moduli = set(m2 for m2, r2 in classes)
                        base_score = sc
                        improved = True
                        break
        
        # 2) Try 1-for-1 swap: remove m_remove, add m_add (m_add < m_remove preferred)
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m_remove = classes[i][0]
            remaining = classes[:i] + classes[i+1:]
            rem_moduli = set(m2 for m2, r2 in remaining)
            for m_add in range(20, 201):
                if m_add in rem_moduli:
                    continue
                valid_rs = _find_valid_residues(m_add, remaining)
                if valid_rs:
                    new_classes = remaining + [[m_add, valid_rs[0]]]
                    if _is_valid(new_classes):
                        sc = _score_fast(new_classes)
                        if sc > base_score + 1e-12:
                            classes = new_classes
                            current_moduli = set(m2 for m2, r2 in classes)
                            base_score = sc
                            improved = True
                            break
            if improved:
                break
        
        # 3) Try 1-for-2 swap: remove m_remove, add two new moduli
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m_remove = classes[i][0]
            remaining = classes[:i] + classes[i+1:]
            rem_moduli = set(m2 for m2, r2 in remaining)
            # Try adding two smaller moduli
            for m1_add in range(20, min(m_remove, 201)):
                if time.time() > deadline:
                    break
                if m1_add in rem_moduli:
                    continue
                valid_rs1 = _find_valid_residues(m1_add, remaining)
                if not valid_rs1:
                    continue
                temp = remaining + [[m1_add, valid_rs1[0]]]
                temp_moduli = set(m2 for m2, r2 in temp)
                for m2_add in range(20, min(m_remove, 201)):
                    if m2_add in temp_moduli or m2_add == m1_add:
                        continue
                    valid_rs2 = _find_valid_residues(m2_add, temp)
                    if valid_rs2:
                        new_classes = temp + [[m2_add, valid_rs2[0]]]
                        if _is_valid(new_classes):
                            sc = _score_fast(new_classes)
                            if sc > base_score + 1e-12:
                                classes = new_classes
                                current_moduli = set(m2 for m2, r2 in classes)
                                base_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        # 4) Try reassigning residues to enable additions
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes
            m, r = classes[i]
            others = classes[:i] + classes[i+1:]
            valid_rs = _find_valid_residues(m, others)
            if len(valid_rs) > 1:
                for new_r in valid_rs:
                    if new_r == r:
                        continue
                    new_classes = classes[:i] + [[m, new_r]] + classes[i+1:]
                    if not _is_valid(new_classes):
                        continue
                    # Check if this allows adding a new modulus
                    nc_moduli = set(m2 for m2, r2 in new_classes)
                    for m_add in range(20, 201):
                        if m_add in nc_moduli:
                            continue
                        valid_add = _find_valid_residues(m_add, new_classes)
                        if valid_add:
                            final_classes = new_classes + [[m_add, valid_add[0]]]
                            if _is_valid(final_classes):
                                sc = _score_fast(final_classes)
                                if sc > base_score + 1e-12:
                                    classes = final_classes
                                    current_moduli = set(m2 for m2, r2 in classes)
                                    base_score = sc
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                break
    
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill
    
    best_classes = None
    best_score = 0.0
    
    # Quick greedy to get an initial solution
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 30:
            break
        valid_rs = _find_valid_residues(m, selected)
        if valid_rs:
            selected.append((m, valid_rs[0]))
    
    if selected:
        classes = [[m, r] for m, r in selected]
        if _is_valid(classes):
            best_score = _score(classes)
            best_classes = classes[:]
    
    # Build conflict graph and run exact branch-and-bound
    if time.time() < deadline - 5:
        try:
            nodes, weights, adj, id_map = _build_conflict_graph()
            n = len(nodes)
            
            # Convert initial solution to node indices
            init_nodes = []
            if best_classes:
                for m, r in best_classes:
                    if (m, r) in id_map:
                        init_nodes.append(id_map[(m, r)])
            
            # Run branch-and-bound with time limit
            bnb_deadline = deadline - 2
            best_w, best_set = _mwis_exact(adj, weights, bnb_deadline, init_nodes)
            
            # Convert back to classes
            if best_set:
                new_classes = []
                tmp = best_set
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    m, r = nodes[v]
                    new_classes.append([m, r])
                if _is_valid(new_classes):
                    sc = _score(new_classes)
                    if sc > best_score:
                        best_score = sc
                        best_classes = new_classes
        except Exception:
            # Fallback to heuristic if B&B fails
            pass
    
    # If B&B didn't improve or timed out, try local search on best
    if best_classes and time.time() < deadline - 1:
        improved = _local_search(best_classes, deadline - 0.5)
        sc = _score(improved)
        if sc > best_score:
            best_score = sc
            best_classes = improved
    
    if best_classes is None:
        # Fallback
        best_classes = [[20, 0], [40, 1], [60, 2], [80, 3], [100, 4],
                       [120, 5], [140, 6], [160, 7], [180, 8], [200, 9]]
    
    return best_classes
