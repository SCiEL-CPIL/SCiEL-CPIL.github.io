"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """
    Maximum independent set search on conflict graph of (residue, modulus) pairs.
    Uses bitset-accelerated candidate filtering with greedy + local search.
    """
    if deadline is None:
        deadline = time.time() + 80  # safety margin
    
    all_moduli = list(range(2, max_mod + 1))
    
    # Pre-compute gcd table
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for m1 in range(2, max_mod + 1):
        for m2 in range(2, max_mod + 1):
            gcd_table[m1][m2] = gcd(m1, m2)
    
    # Build node list: index -> (r, m)
    nodes = []
    for m in all_moduli:
        for r in range(m):
            nodes.append((r, m))
    
    N = len(nodes)
    
    # Build conflict bitsets
    conflict = [0] * N
    for i in range(N):
        r1, m1 = nodes[i]
        for j in range(i + 1, N):
            r2, m2 = nodes[j]
            g = gcd_table[m1][m2]
            if (r1 - r2) % g == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy initial solution
    def greedy_initial():
        available = (1 << N) - 1
        selected = []
        while available:
            best_node = -1
            best_conflicts = N + 1
            avail = available
            while avail:
                lb = avail & (-avail)
                idx = lb.bit_length() - 1
                c = bin(conflict[idx] & available).count('1')
                if c < best_conflicts:
                    best_conflicts = c
                    best_node = idx
                avail ^= lb
            if best_node == -1:
                break
            selected.append(best_node)
            available &= ~(conflict[best_node] | (1 << best_node))
        return selected
    
    best_set = greedy_initial()
    best_size = len(best_set)
    
    # Local improvement with maintained bitset
    def local_improve(selected_nodes, deadline):
        selected = set(selected_nodes)
        sel_bits = 0
        for s in selected:
            sel_bits |= (1 << s)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add one node
            for i in range(N):
                if i in selected:
                    continue
                if conflict[i] & sel_bits:
                    continue
                selected.add(i)
                sel_bits |= (1 << i)
                improved = True
                break
            if improved:
                continue
            # Try 2-exchange: remove one, add two
            for i in list(selected):
                if time.time() > deadline:
                    break
                selected.discard(i)
                sel_bits &= ~(1 << i)
                added = []
                for j in range(N):
                    if j in selected or j == i:
                        continue
                    if conflict[j] & sel_bits:
                        continue
                    selected.add(j)
                    sel_bits |= (1 << j)
                    added.append(j)
                    if len(added) == 2:
                        break
                if len(added) == 2:
                    improved = True
                else:
                    for j in added:
                        selected.discard(j)
                        sel_bits &= ~(1 << j)
                    selected.add(i)
                    sel_bits |= (1 << i)
                if improved:
                    break
        return list(selected)
    
    improved_set = local_improve(best_set, deadline - 5)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set = improved_set
    
    # Convert to output format
    result = []
    for idx in best_set:
        r, m = nodes[idx]
        result.append([r, m])
    
    # Validate
    if not _is_valid_set(result):
        result = [[0, 2], [1, 4], [0, 6], [2, 8], [0, 10], [4, 12], [0, 14]]
    
    return result

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
