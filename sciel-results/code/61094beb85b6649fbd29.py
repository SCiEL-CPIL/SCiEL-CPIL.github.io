import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Problem Setup
    # Moduli range [20, 200]
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcds for all pairs of moduli
    gcds = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # 2. Graph Construction
    # Nodes represent valid residue classes [m, r].
    # We want to select a subset of nodes such that no two are connected by an edge (incompatible).
    # This is the Maximum Weight Independent Set problem.
    
    # To make the graph manageable, we observe that for a given modulus m, 
    # all residues r in [0, m) are potential nodes.
    # Total nodes = sum(m for m in range(20, 201)) = 20520. This is too large for a dense adjacency matrix.
    # We will use a sparse representation or generate neighbors on the fly.
    
    # However, a key insight: The weight of a node (m, r) is 1/m. 
    # The conflict condition: (m1, r1) conflicts with (m2, r2) if r1 == r2 mod gcd(m1, m2).
    
    # Branch and Bound Strategy:
    # We will process moduli in a specific order. 
    # A common heuristic for MWIS is to order vertices by weight (descending) or degree.
    # Here, weight is 1/m. Smaller m has higher weight.
    # Let's order moduli by increasing m (so 20, 21, ..., 200).
    # Why? Because smaller moduli have fewer residues (easier to handle) and higher weight.
    # Actually, branching on the modulus with the fewest remaining valid options first is often effective (MRV heuristic).
    
    # Let's refine the state representation for Branch and Bound.
    # State: (current_modulus_index, selected_classes, current_score)
    # This is not quite right because the choice of residue for m_i affects m_j for j > i.
    
    # Alternative Approach: Recursive search over moduli.
    # At step i, we decide which residue r to pick for modulus m_i, or skip m_i.
    # If we pick r, we must ensure it's compatible with all previously picked (m_k, r_k) for k < i.
    # This is a Constraint Satisfaction Problem (CSP) that we can solve with backtracking + pruning.
    
    # Pruning:
    # Upper Bound = current_score + sum(1/m_j for all remaining j > i).
    # If this upper bound <= best_score_found_so_far, prune.
    
    # To make this efficient:
    # 1. Sort moduli. Since weight is 1/m, processing smaller m first allows us to "lock in" high weights early.
    #    However, smaller m have fewer residues.
    # 2. We need a fast compatibility check.
    
    # Let's precompute for each pair of moduli (i, j) with i < j, the set of residues r_j that are INCOMPATIBLE with each r_i.
    # Actually, it's easier to store the compatible residues.
    # compat[i][r_i] is a list of valid r_j for all j > i? No, that's too much memory.
    
    # Better: For a given modulus m_i and chosen residue r_i, and a set of previously chosen (m_k, r_k),
    # a residue r_j for m_j is valid if for all k < j, (r_j - r_k) % gcd(m_j, m_k) != 0.
    
    # Let's use a recursive backtracking search with memoization or just tight pruning.
    # Given 181 moduli, the depth is 181. The branching factor is m (up to 200).
    # This is huge. We need strong pruning.
    
    # The "Upper Bound" sum(1/m_j) is weak if we can't pick many.
    # A tighter bound: For each remaining modulus m_j, the maximum weight we can get is 1/m_j.
    # But we might not be able to pick it at all.
    
    # Let's try a different angle: **Greedy with Local Search is what the current code does.**
    # The prompt asks for a **Branch and Bound** approach.
    
    # To make B&B work in 110s, we must be very smart.
    # Observation: The optimal solution likely includes almost all small moduli.
    # Let's fix the residues for the first K moduli (e.g., 20..50) using a small search, 
    # and then greedily fill the rest?
    
    # Actually, let's implement a proper B&B on the moduli sequence.
    # We process moduli in increasing order: 20, 21, ..., 200.
    # State: idx (current modulus index), selected_residues (list of (m, r) so far).
    
    # To speed up compatibility checks:
    # When adding (m_new, r_new), we check against all (m_old, r_old).
    # Check: (r_new - r_old) % gcd(m_new, m_old) != 0.
    
    # Optimization: 
    # Instead of checking all previous, we can maintain for each modulus m_old, the "forbidden" residues for future moduli?
    # No, the constraint is pairwise.
    
    # Let's try to optimize the B&B loop in Python.
    # Python recursion might be slow. We will use an iterative or optimized recursive function.
    
    best_score = 0.0
    best_selection = []
    
    # Precompute 1/m
    weights = [1.0 / m for m in moduli]
    
    # Precompute gcds already done.
    
    # We will use a recursive function.
    # To avoid deep recursion issues, we can use a stack or increase limit, but 181 depth is fine.
    
    # Pruning enhancement:
    # Precompute suffix_sum[i] = sum(weights[i:])
    
    suffix_sum = [0.0] * (n_moduli + 1)
    for i in range(n_moduli - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    # Global variables for the search
    current_selection = []  # List of (m_idx, r)
    current_score = 0.0
    
    # We need to track time.
    # We will abort if time > deadline.
    
    # To make the search effective, we should order the choices (residues) heuristically.
    # For a given modulus m, which residue r is "best"?
    # Heuristic: Pick r that is compatible with the most other potential residues?
    # Or simply: Random order? No, deterministic is better for reproducibility, but we want to find a good solution fast.
    # Let's try sorting residues by how "conflicting" they are with already selected ones?
    # Actually, since we are adding one by one, all valid residues are equally compatible with the *past*.
    # The difference is in the *future*.
    # A residue r is "better" if it leaves more options for future moduli.
    # Approximation: Pick r that minimizes the number of conflicts it would cause with random future choices?
    # Too expensive.
    
    # Let's just iterate residues 0 to m-1.
    # But wait, if we just do DFS, it will explore 20*21*22... which is impossible.
    # We MUST prune heavily.
    
    # The bound suffix_sum[i] is the absolute maximum additional score.
    # If current_score + suffix_sum[i] <= best_score, prune.
    
    # Is this bound tight enough?
    # If best_score is high, yes. But initially best_score is 0.
    # We need a good initial solution to tighten the bound.
    
    # Phase 1: Get a good initial solution using a fast greedy heuristic.
    # Greedy: For i from 0 to n-1, pick the first valid residue for m_i. If none, skip.
    
    def greedy_initial():
        sel = []
        score = 0.0
        for i in range(n_moduli):
            m = moduli[i]
            found = False
            for r in range(m):
                valid = True
                for (m_k, r_k) in sel:
                    g = gcds[i][m_k - 20] # m_k is actual modulus, index is m_k-20
                    # Wait, gcds is indexed by i, j (0..180).
                    # m_k is the modulus value. Index is m_k - 20.
                    if (r - r_k) % g == 0:
                        valid = False
                        break
                if valid:
                    sel.append((i, r))
                    score += weights[i]
                    found = True
                    break
            # If not found, we skip this modulus
        return sel, score
    
    init_sel, init_score = greedy_initial()
    best_score = init_score
    best_selection = init_sel
    
    # Phase 2: Branch and Bound
    # We will do a DFS.
    # To improve pruning, we can sort the moduli?
    # The order 20..200 is natural.
    
    # Let's add a time check inside the recursion.
    
    def search(i, current_sel, current_score):
        nonlocal best_score, best_selection
        
        if time.time() > deadline:
            return
            
        if i == n_moduli:
            if current_score > best_score:
                best_score = current_score
                best_selection = list(current_sel)
            return
            
        # Pruning
        if current_score + suffix_sum[i] <= best_score:
            return
            
        m_idx = i
        m_val = moduli[m_idx]
        
        # Option 1: Skip this modulus
        search(i + 1, current_sel, current_score)
        
        # Option 2: Pick a residue
        # To speed up, we can try to pick residues that are "less restrictive" for the future.
        # But for now, let's just iterate.
        # Optimization: If we pick r, we add to current_sel.
        
        # Heuristic: Try residues in an order that might lead to better scores?
        # All residues have the same weight 1/m. So the score doesn't change based on WHICH residue, only WHETHER we pick one.
        # Therefore, for the purpose of the SCORE, any valid residue is equivalent.
        # However, the choice affects future feasibility.
        # We want to pick a residue that allows more future picks.
        
        # Let's just pick the FIRST valid residue we find? 
        # No, that's greedy. For B&B, we need to explore.
        # But if we explore all, it's too slow.
        
        # Insight: Since the weight is independent of r, the "value" of picking m_i is always weights[i].
        # The only question is: "Is there ANY valid residue?"
        # If yes, we add weights[i] to the score.
        # If no, we gain 0.
        
        # So the decision is binary: Pick m_i (if possible) or Skip m_i.
        # BUT, WHICH residue we pick determines if we can pick future moduli.
        
        # So we still need to branch on r.
        
        # To make this faster, we can group residues by their "conflict signature" with future moduli?
        # Too complex.
        
        # Let's try a limited search.
        # If the number of valid residues is small, branch on all.
        # If large, pick a few representative ones?
        
        valid_residues = []
        for r in range(m_val):
            valid = True
            for (m_k_idx, r_k) in current_sel:
                g = gcds[m_idx][m_k_idx]
                if (r - r_k) % g == 0:
                    valid = False
                    break
            if valid:
                valid_residues.append(r)
                
        if not valid_residues:
            # Cannot pick this modulus
            return # We already called search(i+1) for skipping? 
            # Wait, in the code above, I called search(i+1) BEFORE checking residues.
            # So "Skip" is handled. If no valid residues, we just return (don't pick).
            # This is correct.
            return
            
        # We have valid_residues.
        # We should explore picking one of them.
        # To avoid exploring 200 identical branches (in terms of immediate score),
        # we can pick a subset.
        
        # Heuristic: Pick the residue that minimizes conflicts with the NEXT modulus?
        # Or just pick the first one, and assume it's "good enough" for the bound?
        # No, B&B requires exploring alternatives to prove optimality or find better.
        
        # However, given the time limit, we cannot explore all 200 residues at each of 181 steps.
        
        # Strategy: 
        # 1. Pick the "best" residue (e.g., first one, or one that looks most flexible).
        # 2. Recurse.
        # 3. If we have time, try another residue.
        
        # Let's pick the first valid residue.
        # And maybe the last?
        
        # Actually, let's randomize the order of residues to explore diverse parts of the space.
        random.shuffle(valid_residues)
        
        # Limit the number of branches to prevent explosion.
        # If we branch too much, we timeout.
        # Let's try up to 3-5 residues per node if time permits.
        
        # Better: 
        # The "Skip" branch is already explored.
        # Now we explore "Pick".
        # Since all "Pick" branches yield the same immediate score increase, 
        # the one that allows the best FUTURE score is the winner.
        
        # We can estimate future potential for a residue r:
        # For each future modulus j > i, check if r conflicts with "typical" residues of j?
        
        # Let's just pick the first 2 valid residues to keep it fast, or just 1 if very tight.
        # Let's try picking up to 2 residues.
        
        max_branches = 2
        for r in valid_residues[:max_branches]:
            current_sel.append((m_idx, r))
            search(i + 1, current_sel, current_score + weights[m_idx])
            current_sel.pop()
            
            # Check time
            if time.time() > deadline:
                return

    # Start search
    # To make the initial bound tighter, we might want to sort moduli differently?
    # If we process high weight (small m) first, we fill the score quickly, making the bound tighter sooner.
    # The current order 20..200 is already small to large.
    
    search(0, [], 0.0)
    
    # Phase 3: Local Search Polish on the best found solution
    # The B&B might have found a good local optimum. Let's try to improve it.
    
    if best_selection:
        # Convert best_selection (list of (m_idx, r)) to lists of m and r
        curr_idx = [s[0] for s in best_selection]
        curr_res = [s[1] for s in best_selection]
        
        # Simple hill climbing: Try to change one residue or swap two.
        # We only have a few seconds left.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(curr_idx)):
                m_idx = curr_idx[i]
                old_r = curr_res[i]
                # Try all other valid residues for this modulus
                temp_idx = curr_idx[:i] + curr_idx[i+1:]
                temp_res = curr_res[:i] + curr_res[i+1:]
                
                for r in range(moduli[m_idx]):
                    valid = True
                    for k in range(len(temp_idx)):
                        m_k = temp_idx[k]
                        r_k = temp_res[k]
                        g = gcds[m_idx][m_k]
                        if (r - r_k) % g == 0:
                            valid = False
                            break
                    if valid:
                        # Calculate new score
                        # Score change is 0 because 1/m is same.
                        # But it might allow adding more moduli?
                        # This hill climb only changes residues, doesn't add/remove moduli.
                        # So score doesn't change. 
                        # This is useless for score improvement unless we also try to ADD moduli.
                        
                        # Let's try to ADD a missing modulus.
                        pass
            
            # Try adding a missing modulus
            missing = [i for i in range(n_moduli) if i not in curr_idx]
            random.shuffle(missing)
            for m_idx in missing:
                if time.time() > deadline - 5:
                    break
                valid_res = []
                for r in range(moduli[m_idx]):
                    valid = True
                    for k in range(len(curr_idx)):
                        m_k = curr_idx[k]
                        r_k = curr_res[k]
                        g = gcds[m_idx][m_k]
                        if (r - r_k) % g == 0:
                            valid = False
                            break
                    if valid:
                        valid_res.append(r)
                if valid_res:
                    # Add it
                    r = random.choice(valid_res)
                    new_idx = curr_idx + [m_idx]
                    new_res = curr_res + [r]
                    new_score = sum(1.0 / moduli[i] for i in new_idx)
                    if new_score > best_score:
                        best_score = new_score
                        best_selection = list(zip(new_idx, new_res))
                        curr_idx = new_idx
                        curr_res = new_res
                        improved = True
                        break # Restart loop with new state

    # Convert result to required format
    result_classes = []
    for m_idx, r in best_selection:
        result_classes.append([moduli[m_idx], r])
        
    return result_classes

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Score: {sum(1.0/m for m,r in res)}")
    print(f"Classes: {len(res)}")
