"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)  # 127 moduli
    
    # Precompute: for each pair of moduli (i, j), can we find residues r_i, r_j
    # such that r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)?
    # Two residue classes (r1 mod m1) and (r2 mod m2) are disjoint iff
    # r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    # So for a given pair of moduli, we need to check if there EXISTS a valid
    # residue assignment. This is always possible unless the moduli are equal
    # (which they're not since they're distinct). Wait - actually for any two
    # distinct moduli m1, m2, we can always find r1, r2 such that
    # r1 % gcd(m1,m2) != r2 % gcd(m1,m2) as long as gcd(m1,m2) >= 2.
    # If gcd(m1,m2) = 1, then r1 % 1 = 0 and r2 % 1 = 0 always, so they're
    # NEVER disjoint! That means coprime moduli can NEVER both be in our set.
    
    # Wait, let me re-read: "Every two infinite residue classes must be disjoint"
    # Two classes r1 mod m1 and r2 mod m2 are disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    # If gcd(m1,m2) = 1, then both sides are 0, so they're NEVER disjoint.
    # Therefore, we can only include moduli that are pairwise non-coprime!
    
    # This is a much simpler problem: find the largest subset of {2,...,128}
    # where every pair has gcd > 1.
    
    # Let's verify: if gcd(m1,m2) = 1, then for any r1, r2:
    # r1 % 1 = 0 and r2 % 1 = 0, so r1 % gcd = r2 % gcd, meaning NOT disjoint.
    # So coprime pairs are forbidden.
    
    # If gcd(m1,m2) = g >= 2, we need r1 % g != r2 % g. Since g >= 2, there
    # are g possible residues mod g, so we can always choose r1 and r2 with
    # different residues mod g. Thus non-coprime pairs are always compatible.
    
    # So the problem reduces to: find the maximum subset of {2,...,128} such that
    # every pair has gcd > 1.
    
    # This is a maximum clique problem in the graph where edges connect non-coprime pairs,
    # or equivalently, a maximum independent set in the graph where edges connect coprime pairs.
    
    # Let's build the coprime graph and find the maximum independent set.
    
    coprime_adj = [0] * n  # bitsets
    
    for i in range(n):
        mi = all_moduli[i]
        for j in range(i + 1, n):
            mj = all_moduli[j]
            if gcd(mi, mj) == 1:
                coprime_adj[i] |= (1 << j)
                coprime_adj[j] |= (1 << i)
    
    # Now find maximum independent set in this graph using branch and bound.
    # A set is independent if no two vertices are connected by an edge.
    # Equivalently, for each vertex v in the set, none of its neighbors are in the set.
    
    best_size = 0
    best_set = []
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, start_time, deadline
        
        if time.time() > deadline - 2:
            return
        
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick the vertex with the most connections within candidates
        v_idx = -1
        v_score = -1
        cand_list = list(candidates)
        for idx in cand_list:
            # Count neighbors of idx within candidates
            count = 0
            for other in cand_list:
                if other != idx and (coprime_adj[idx] >> other) & 1:
                    count += 1
            if count > v_score:
                v_score = count
                v_idx = idx
        
        if v_idx == -1:
            # No candidates
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        v = v_idx
        
        # Option 1: Include v
        # Remove v and all its coprime neighbors from candidates
        new_candidates = []
        for c in candidates:
            if c == v:
                continue
            if (coprime_adj[v] >> c) & 1:
                continue
            new_candidates.append(c)
        
        current_set.append(v)
        branch_and_bound(new_candidates, current_set, current_size + 1)
        current_set.pop()
        
        # Option 2: Exclude v
        new_candidates2 = [c for c in candidates if c != v]
        branch_and_bound(new_candidates2, current_set, current_size)
    
    # Initial candidates: all moduli
    all_candidates = list(range(n))
    branch_and_bound(all_candidates, [], 0)
    
    if best_size > 0:
        # Now assign residues: for each selected modulus, we need to find a residue
        # such that for every other selected modulus, the residues are compatible.
        # Since all pairs have gcd > 1, we just need to ensure pairwise compatibility.
        
        selected_moduli = [all_moduli[i] for i in best_set]
        
        # Greedily assign residues
        result = []
        for m in selected_moduli:
            found = False
            for r in range(m):
                compatible = True
                for r2, m2 in result:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    result.append((r, m))
                    found = True
                    break
            if not found:
                # Should not happen if our logic is correct, but handle gracefully
                # Try all residues again more carefully
                for r in range(m):
                    compatible = True
                    for r2, m2 in result:
                        g = gcd(m, m2)
                        if r % g == r2 % g:
                            compatible = False
                            break
                    if compatible:
                        result.append((r, m))
                        break
        
        return [[r, m] for r, m in result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
