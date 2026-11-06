import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    # Generate all candidate pairs (m, r)
    # m in [20, 200], r in [0, m)
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    
    n = len(candidates)
    
    # Precompute conflict graph
    # Two pairs (m1, r1) and (m2, r2) conflict if:
    # 1. They have the same modulus m1 == m2 (since moduli must be distinct)
    # 2. They are not disjoint: r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # For efficiency, we can store conflicts as a list of bitsets or just a list of lists
    # Since n is around 20*10 + 21*11 + ... ~ 20000, a list of lists is fine.
    # Actually, sum of m for m=20..200 is approx 200*110 - 210 = 21790.
    # 21790^2 is ~ 470 million operations for precomputation. This might be too slow in Python.
    
    # Optimization: 
    # Conflict condition 1: Same modulus.
    # Conflict condition 2: Same residue modulo gcd.
    
    # Let's group candidates by modulus.
    # For each pair of moduli (m1, m2), we can determine which residues conflict.
    
    # However, building the full adjacency list for ~22k nodes might be heavy.
    # Let's try a smarter approach for the Branch and Bound.
    
    # We want to maximize sum(1/m).
    # Sort candidates by weight (1/m) descending.
    # This helps the branch and bound find a good solution quickly.
    
    weights = [1.0 / m for m, r in candidates]
    
    # Sort candidates by weight descending
    # If weights are equal, sort by m ascending (smaller m first? doesn't matter much)
    indices = list(range(n))
    indices.sort(key=lambda i: -weights[i])
    
    sorted_candidates = [candidates[i] for i in indices]
    sorted_weights = [weights[i] for i in indices]
    
    # We need a fast conflict check.
    # Let's build the conflict graph as a list of sets of indices.
    # But 22k nodes * 22k nodes is too big for a dense matrix or even list of lists if not careful.
    
    # Alternative: Use a bitset representation for the current selection to check conflicts fast?
    # Or, since the graph is relatively sparse (conflicts are specific), maybe we can generate conflicts on the fly?
    
    # Let's try to build the adjacency list efficiently.
    # For each pair of moduli m1, m2:
    #   g = gcd(m1, m2)
    #   For each r1 in [0, m1) and r2 in [0, m2):
    #     if r1 % g == r2 % g, they conflict.
    
    # We can precompute for each modulus m, and each residue r, the list of (m2, r2) that conflict with it.
    # This is still potentially large.
    
    # Let's estimate the density.
    # For m1, m2, the number of conflicting pairs is sum_{k=0}^{g-1} (count_r1(k) * count_r2(k)).
    # count_r(k) is either floor(m/g) or ceil(m/g).
    # So density is roughly 1/g.
    # Average gcd is small, so density is low?
    # Actually, for m1=20, m2=20, g=20. All residues conflict with all residues of same modulus?
    # No, if m1==m2, they conflict if r1==r2? No, the constraint is "distinct moduli".
    # So if m1==m2, they ALWAYS conflict because we can only pick one residue per modulus?
    # Wait, the problem says "distinct moduli". So we can only pick ONE pair for a given modulus.
    # Therefore, all pairs with the same modulus conflict with each other.
    
    # So, Conflict(m1, r1, m2, r2) is True if:
    # 1. m1 == m2
    # 2. m1 != m2 and r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # This structure is helpful.
    
    # Let's build the graph.
    # adj[i] = list of indices j such that i and j conflict.
    
    # To speed up, we can iterate by moduli.
    # Map modulus to list of candidate indices.
    mod_to_indices = {}
    for i, (m, r) in enumerate(sorted_candidates):
        if m not in mod_to_indices:
            mod_to_indices[m] = []
        mod_to_indices[m].append(i)
    
    adj = [[] for _ in range(n)]
    
    # 1. Add conflicts within same modulus
    for m, idxs in mod_to_indices.items():
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                adj[idxs[i]].append(idxs[j])
                adj[idxs[j]].append(idxs[i])
                
    # 2. Add conflicts between different moduli
    # Iterate over pairs of moduli
    moduli_list = sorted(mod_to_indices.keys())
    num_moduli = len(moduli_list)
    
    for i_mod in range(num_moduli):
        m1 = moduli_list[i_mod]
        idxs1 = mod_to_indices[m1]
        for j_mod in range(i_mod + 1, num_moduli):
            m2 = moduli_list[j_mod]
            idxs2 = mod_to_indices[m2]
            g = gcd(m1, m2)
            
            # For each r1, r2, check conflict
            # To optimize, we can group residues by r % g
            # res_class_to_indices1 = defaultdict(list)
            # for idx in idxs1:
            #    r1 = sorted_candidates[idx][1]
            #    res_class_to_indices1[r1 % g].append(idx)
            # Similarly for 2.
            # Then for each class k, all pairs (i1, i2) where i1 in class1[k] and i2 in class2[k] conflict.
            
            rc1 = {}
            for idx in idxs1:
                r1 = sorted_candidates[idx][1]
                key = r1 % g
                if key not in rc1:
                    rc1[key] = []
                rc1[key].append(idx)
                
            rc2 = {}
            for idx in idxs2:
                r2 = sorted_candidates[idx][1]
                key = r2 % g
                if key not in rc2:
                    rc2[key] = []
                rc2[key].append(idx)
                
            for key in rc1:
                if key in rc2:
                    for i1 in rc1[key]:
                        for i2 in rc2[key]:
                            adj[i1].append(i2)
                            adj[i2].append(i1)
                            
    # Now we have the conflict graph.
    # We want to find Max Weight Independent Set.
    # This is equivalent to Max Weight Clique in the complement graph, or just MWIS on this graph.
    
    # Branch and Bound for MWIS:
    # State: current set of selected nodes, next node to consider.
    # Since nodes are sorted by weight, we can use a standard B&B.
    
    # However, n is ~22k. A naive B&B might be too slow.
    # We need a good upper bound.
    # Upper bound: Sum of weights of all remaining nodes that are compatible with the current selection?
    # Or simpler: Sum of weights of all remaining nodes (ignoring conflicts).
    # This is a weak bound.
    
    # Better bound: For the remaining nodes, find a maximal independent set greedily?
    # Or use the fact that we can't pick more than one node from each connected component of the conflict graph?
    # No, that's not an upper bound on weight.
    
    # Let's try a heuristic first to get a good lower bound (incumbent).
    # Greedy: Pick node with max weight, remove it and its neighbors, repeat.
    
    def greedy_mwis():
        remaining = set(range(n))
        selected = []
        total_weight = 0.0
        while remaining:
            # Find max weight node in remaining
            max_w = -1
            max_node = -1
            for i in remaining:
                if sorted_weights[i] > max_w:
                    max_w = sorted_weights[i]
                    max_node = i
            if max_node == -1:
                break
            selected.append(max_node)
            total_weight += max_w
            # Remove max_node and its neighbors from remaining
            to_remove = {max_node}
            for neighbor in adj[max_node]:
                if neighbor in remaining:
                    to_remove.add(neighbor)
            remaining -= to_remove
        return selected, total_weight
    
    best_nodes, best_score = greedy_mwis()
    
    # Now B&B
    # To make B&B efficient, we can use a recursive approach with memoization or pruning.
    # But n=22k is large.
    # Maybe we can reduce the problem size?
    # Note that many moduli are even.
    # Let's stick to the full set but optimize the B&B.
    
    # Alternative: Since the graph is sparse, maybe we can use a different algorithm?
    # But B&B with a good incumbent is often the best for MWIS.
    
    # Let's implement a B&B that explores nodes in order of weight.
    # We maintain a list of "available" nodes.
    
    # To speed up, we can precompute for each node, the list of neighbors that have higher weight?
    # No, the order is fixed.
    
    # Let's try a simpler B&B:
    # f(idx, current_weight, current_set)
    # idx: index in sorted_candidates
    # If idx == n, update best.
    # Upper bound: current_weight + sum(sorted_weights[idx:])
    # If upper_bound <= best_score, prune.
    
    # This upper bound is very loose.
    # We can improve it by only summing weights of nodes that are compatible with current_set?
    # That requires checking compatibility, which is O(n) per step.
    
    # Given the time limit, maybe the greedy solution is already very good?
    # The greedy solution picks the highest weight nodes.
    # The conflicts are local.
    
    # Let's try to improve the greedy solution by local search?
    # Or just run the B&B for a limited depth?
    
    # Actually, let's look at the structure again.
    # The problem is equivalent to finding a set of moduli and assigning residues.
    # The greedy approach on moduli (picking smallest moduli first) might be better?
    # No, we sorted by 1/m, which is same as sorting by m ascending.
    # So the greedy picks m=20, then m=21, etc.
    
    # Let's check if the greedy solution is optimal or near-optimal.
    # The B&B might not finish in 150s for n=22k.
    
    # Let's try a different strategy:
    # Since the moduli are distinct, we are effectively choosing a subset of moduli.
    # For a chosen subset of moduli, is it always possible to assign residues?
    # No, the residues must be pairwise disjoint.
    # This is a system of congruences.
    # r_i != r_j mod gcd(m_i, m_j).
    
    # This is a constraint satisfaction problem.
    # We can use backtracking on the moduli.
    # Sort moduli by m ascending.
    # For each modulus, try all residues.
    # Check consistency with previous moduli.
    
    # This is what the current program does (greedy).
    # The greedy fails if it picks a residue that blocks a future high-value modulus?
    # No, high-value moduli are small. So we process small moduli first.
    # So the greedy picks residues for small moduli.
    # If a small modulus residue blocks a larger modulus, that's fine because larger modulus has lower weight.
    # But what if a different residue for the small modulus would allow MORE larger moduli?
    # Since we want to maximize sum(1/m), and 1/m decreases, we prefer to include as many small moduli as possible.
    # The greedy includes ALL small moduli if possible.
    # It only fails to include a modulus if NO residue works.
    
    # So the greedy solution includes a set of moduli S.
    # Can we add more moduli?
    # The greedy tries to add m in ascending order.
    # If it fails to add m, it skips it.
    # Maybe a different residue choice for a previous modulus would allow m to be added?
    
    # This suggests a local search or backtracking on the residue choices.
    
    # Let's implement a backtracking search with pruning.
    # Order moduli by m ascending.
    # State: list of (m, r) chosen so far.
    # For next modulus m, try all r.
    # Check compatibility.
    
    # To prune, we can use an upper bound:
    # Current score + sum(1/m' for all m' > m that are not yet decided and are compatible with current set?)
    # This is hard to compute fast.
    
    # Let's try a simple backtracking with a time limit.
    
    moduli_sorted = list(range(20, 201))
    
    best_solution = []
    best_score = 0.0
    
    def backtrack(idx, current_selection, current_score):
        nonlocal best_score, best_solution
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if idx == len(moduli_sorted):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_selection)
            return
            
        m = moduli_sorted[idx]
        
        # Try all residues
        for r in range(m):
            # Check compatibility
            valid = True
            for m_sel, r_sel in current_selection:
                g = gcd(m, m_sel)
                if r % g == r_sel % g:
                    valid = False
                    break
            
            if valid:
                current_selection.append((m, r))
                backtrack(idx + 1, current_selection, current_score + 1.0/m)
                current_selection.pop()
                
            # Pruning?
            # If we skip m, can we still beat best_score?
            # Upper bound: current_score + sum(1/m' for m' in moduli_sorted[idx+1:])
            # This is a weak bound.
            # But we can precompute suffix sums of 1/m.
            
        # Also try skipping m
        # But we must check if skipping m allows a better solution?
        # Actually, if we can include m, we should, because 1/m > 0.
        # Unless including m blocks many other moduli?
        # But we process in ascending order. So m is the smallest remaining.
        # Including m might block larger moduli.
        # But 1/m is larger than any 1/m' for m' > m.
        # So generally, we want to include m.
        # However, if including m with ANY residue blocks a lot of future moduli, maybe skipping m is better?
        # Example: m=20. If we pick r=0, it might block m=40, 60, etc.
        # If we pick r=1, it might block different ones.
        # If we skip m=20, we lose 1/20.
        # Can we gain more than 1/20 from unblocking?
        # Unblocking m=40 gives 1/40. Unblocking m=60 gives 1/60.
        # Sum of 1/m' for m' > 20 is finite.
        # Sum_{m=21}^{200} 1/m ~ ln(200) - ln(20) ~ 2.3.
        # So it's possible that skipping a small modulus allows many larger ones.
        # But the greedy already tries to include all.
        # The backtracking will explore skipping if it leads to a better score.
        
        # We need to explore both "include m with some r" and "skip m".
        # But "include m" has m branches.
        # "Skip m" has 1 branch.
        
        # Let's add the skip branch.
        # But we must ensure we don't double count.
        # The loop above handles "include m".
        # We need to call backtrack(idx+1, current_selection, current_score) for skipping.
        
        # However, if we can include m, is it ever optimal to skip it?
        # Yes, if the residue constraints are tight.
        
        # Let's add the skip call.
        # But we should only skip if including m doesn't lead to a better solution?
        # No, we must explore both.
        
        # To avoid explosion, we can use the upper bound.
        # Precompute suffix sums.
        
    # Precompute suffix sums of 1/m
    suffix_sums = [0.0] * (len(moduli_sorted) + 1)
    for i in range(len(moduli_sorted) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i+1] + 1.0 / moduli_sorted[i]
        
    # Modify backtrack to use pruning
    def backtrack_pruned(idx, current_selection, current_score):
        nonlocal best_score, best_solution
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        # Upper bound
        ub = current_score + suffix_sums[idx]
        if ub <= best_score:
            return
            
        if idx == len(moduli_sorted):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_selection)
            return
            
        m = moduli_sorted[idx]
        
        # Try all residues
        for r in range(m):
            valid = True
            for m_sel, r_sel in current_selection:
                g = gcd(m, m_sel)
                if r % g == r_sel % g:
                    valid = False
                    break
            
            if valid:
                current_selection.append((m, r))
                backtrack_pruned(idx + 1, current_selection, current_score + 1.0/m)
                current_selection.pop()
                
        # Skip m
        backtrack_pruned(idx + 1, current_selection, current_score)
        
    # Initialize with greedy solution to set a good best_score
    # The greedy solution from before is a good incumbent.
    # Let's use it.
    if best_score == 0.0:
         # Use the greedy result from the MWIS attempt
         # But that was on candidates, not moduli.
         # Let's just run a simple greedy on moduli to get an initial best.
         greedy_sel = []
         greedy_score = 0.0
         for m in moduli_sorted:
             for r in range(m):
                 valid = True
                 for m_sel, r_sel in greedy_sel:
                     g = gcd(m, m_sel)
                     if r % g == r_sel % g:
                         valid = False
                         break
                 if valid:
                     greedy_sel.append((m, r))
                     greedy_score += 1.0/m
                     break
         best_score = greedy_score
         best_solution = greedy_sel
         
    backtrack_pruned(0, [], 0.0)
    
    # Return the best solution found
    if not best_solution:
        # Fallback
        return [[20, 0]]
        
    return [[m, r] for m, r in best_solution]

if __name__ == "__main__":
    res = construct()
    print(res)
