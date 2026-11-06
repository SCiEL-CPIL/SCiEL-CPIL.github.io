"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    N = 128
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # Precompute conflict counts: 
    # conflicts[a][b] = number of c in [1..N] such that {a,b,c} forms a bad triangle
    # A bad triangle is when gcd(a,b) == gcd(a,c) == gcd(b,c)
    # For a fixed pair (a,b) with g = gcd(a,b), a third element c creates a bad triangle
    # iff gcd(a,c) == g and gcd(b,c) == g.
    # This implies c is a multiple of g, and c/g is coprime to a/g and b/g? 
    # Actually, gcd(a,c)=g means g|c and gcd(a/g, c/g)=1.
    # Similarly for b.
    # So c = g*k. We need gcd(a/g, k)=1 and gcd(b/g, k)=1.
    # Let A = a/g, B = b/g. We need k such that gcd(A,k)=1 and gcd(B,k)=1.
    # This is equivalent to gcd(A*B, k)=1? Not exactly, but close.
    # Actually, the set of forbidden k are those sharing a factor with A or B.
    
    # We will precompute for each pair (a,b) the list of c's that form a bad triangle.
    # But storing lists is memory heavy. 
    # Instead, for Simulated Annealing, we need to know:
    # 1. If we add x to set S, how many new bad triangles are created?
    #    This is the number of pairs (a,b) in S such that {a,b,x} is a bad triangle.
    #    Let conflict_count[x][S] be this number.
    #    We can maintain an array `conflict_counts` of size N+1, where conflict_counts[x] 
    #    is the number of bad triangles involving x and two other elements in the current set.
    
    # To update conflict_counts efficiently when adding/removing an element:
    # When adding x:
    #   For each y in S:
    #     For each z in S (z != y):
    #       If {x,y,z} is bad, increment conflict_counts[x], conflict_counts[y], conflict_counts[z].
    # This is O(|S|^2) per add, which is too slow for SA with many steps.
    
    # Alternative: Precompute a bitmask or count for each pair (a,b) of how many c's make it bad?
    # No, we need the count relative to the *current* set.
    
    # Let's use a different metric: 
    # Let E(S) be the number of bad triangles in S.
    # We want to minimize E(S) to 0.
    # When adding x, delta_E = number of pairs (a,b) in S such that {a,b,x} is bad.
    # Let's precompute for each x, a list of pairs (a,b) that form a bad triangle with x.
    # bad_pairs_with[x] = list of (a,b) such that gcd(a,b)=gcd(a,x)=gcd(b,x).
    
    bad_pairs_with = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            g_ax = gcd_table[a][x]
            if g_ax == 1: # If coprime, gcd(a,b) must be 1, gcd(b,x) must be 1.
                # Then b must be coprime to both a and x.
                # This is getting complex. Let's just iterate b.
                pass
            for b in range(a + 1, N + 1):
                if b == x: continue
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    bad_pairs_with[x].append((a, b))
                    # We store (a,b) for x. For symmetry, we could store x in bad_pairs_with[a] and bad_pairs_with[b]?
                    # But for adding x, we only care about pairs in S.
                    # So bad_pairs_with[x] is sufficient to check delta_E when adding x.
                    # We just need to count how many of these pairs (a,b) are both in S.
                    
    # To make this O(1) or O(log N) per check:
    # We can maintain a bitmask of the current set S.
    # For each x, we have a list of pairs. Checking if a pair is in S requires checking 2 bits.
    # If the list is long, this is slow.
    
    # Let's estimate the length of bad_pairs_with[x].
    # Worst case x=1. gcd(1,a)=1, gcd(1,b)=1. Bad if gcd(a,b)=1.
    # Number of coprime pairs in [1,128] is roughly 6/pi^2 * 128^2 ~ 8000.
    # So for x=1, the list has ~8000 entries.
    # Checking 8000 entries per add is too slow for 100k+ SA steps.
    
    # Optimization:
    # Group pairs by their first element?
    # Or, precompute for each x, a bitmask of "conflicting neighbors"?
    # No, the conflict is pairwise.
    
    # Let's try a different angle:
    # Instead of counting all bad triangles, we can just check validity incrementally.
    # But SA needs fast evaluation.
    
    # What if we use a "conflict graph" where edges represent bad pairs?
    # No, a bad triangle is a hyperedge.
    
    # Let's stick to the O(|S|^2) check but optimize the inner loop with bitmasks?
    # Or, precompute for each x, a list of "potential conflicts" and use a counter?
    
    # Let's try a hybrid:
    # 1. Generate a good initial set using the previous greedy method (fast).
    # 2. Use SA to improve it.
    # In SA, we try to add a random x not in S, or remove a random x in S.
    # To evaluate adding x:
    #   We need to know if it creates any bad triangle.
    #   We can precompute for each x, a list of pairs (a,b) that form a bad triangle with x.
    #   Then we count how many of these pairs are in S.
    #   If count > 0, the move is "bad" (increases E).
    #   If count == 0, the move is "good" (E stays 0 if it was 0, or decreases if it was >0? No, adding x only increases E).
    #   Actually, if S is valid (E=0), adding x is valid iff count == 0.
    #   If S is invalid (E>0), we might still add x if it doesn't make it worse?
    #   Usually SA tries to minimize E.
    
    # Let's refine the precomputation for speed.
    # For each x, store a list of pairs (a,b).
    # To count how many are in S, we iterate the list and check bitmask.
    # If the list is large, this is slow.
    
    # Can we reduce the list size?
    # For x, the pairs (a,b) must satisfy gcd(a,b)=gcd(a,x)=gcd(b,x)=g.
    # This implies a,b,x are all multiples of g.
    # Let a=g*a', b=g*b', x=g*x'.
    # gcd(a',b')=gcd(a',x')=gcd(b',x')=1.
    # So a',b',x' are pairwise coprime.
    
    # Let's just use the list and hope it's fast enough in C (NumPy?) or optimized Python.
    # Python loops are slow.
    
    # Alternative: Use a different objective.
    # Maximize size, subject to validity.
    # This is a Max-Independent-Set-like problem on a 3-uniform hypergraph.
    
    # Let's try a simpler heuristic first:
    # 1. Greedy construction with multiple restarts (current approach is decent).
    # 2. Improve with 2-opt swaps: remove two, add two?
    
    # Let's try to implement the SA with the list of pairs, but limit the list size or use a smarter check.
    # Actually, if we maintain the set as a Python set, checking `a in S and b in S` is O(1) average.
    # Iterating 8000 pairs is ~8000 operations.
    # 10,000 steps * 8000 ops = 80M ops. In Python, this is ~10-20 seconds.
    # This is feasible within 90s if we are careful.
    
    # Let's implement SA.
    
    # 1. Precompute bad_pairs_with[x] as a list of (a,b).
    #    To save memory and time, we can store just the pairs.
    
    # 2. Initial set: Use a greedy construction to get a large valid set.
    
    # 3. SA Loop:
    #    - Current set S (as a set and a bitmask for fast checking? Set is fine for membership).
    #    - Temperature T.
    #    - Move: Pick a random x not in S (add) or in S (remove).
    #      Or pick a random x in S and a random y not in S (swap).
    #    - Calculate delta_E.
    #      If adding x: delta_E = number of pairs (a,b) in bad_pairs_with[x] such that a in S and b in S.
    #      If removing x: delta_E = - (number of pairs (a,b) in bad_pairs_with[x] such that a in S and b in S).
    #      Note: Removing x reduces E by exactly the number of bad triangles involving x.
    #    - Accept if delta_E < 0 or random() < exp(-delta_E / T).
    
    # Let's optimize the "count pairs in S" step.
    # We can precompute for each x, a list of pairs.
    # We can also precompute for each x, a list of "neighbors" a such that there exists b with (a,b) in bad_pairs_with[x].
    # But we need to check pairs.
    
    # Let's try to speed up the pair checking.
    # Instead of a list of tuples, we can use a bitmask for the "other" elements?
    # No, pairs are specific.
    
    # Let's just run it and see if it's fast enough.
    
    # Precompute bad_pairs_with
    bad_pairs_with = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x: continue
            g_ax = gcd_table[a][x]
            for b in range(a + 1, N + 1):
                if b == x: continue
                g_ab = gcd_table[a][b]
                g_bx = gcd_table[b][x]
                if g_ab == g_ax == g_bx:
                    bad_pairs_with[x].append((a, b))
                    
    # Greedy initial construction
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for x in order:
            valid = True
            for a in selected:
                g_ax = gcd_table[a][x]
                for b in selected:
                    if b == a: continue
                    g_ab = gcd_table[a][b]
                    if g_ab == g_ax:
                        g_bx = gcd_table[b][x]
                        if g_bx == g_ab:
                            valid = False
                            break
                if not valid: break
            if valid:
                selected.append(x)
                selected_set.add(x)
        return selected, selected_set

    # Generate a good starting set
    best_set = []
    best_set_set = set()
    
    # Try a few greedy orders
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        s, ss = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            best_set_set = ss
            
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Simulated Annealing
    # We want to maximize size.
    # So our objective is: Maximize |S| subject to E(S)=0.
    # If E(S)>0, we penalize it heavily.
    # Score = |S| - 1000 * E(S).
    
    # Let's define a function to get the score.
    # But calculating E(S) from scratch is O(N^3).
    # We maintain E(S) incrementally.
    
    # Initial E(S) should be 0 since we started with a valid set.
    current_E = 0
    
    # To update E(S) when adding/removing x, we need the count of bad triangles involving x.
    # Let's write a helper function.
    
    def count_conflicts(x, S):
        count = 0
        for a, b in bad_pairs_with[x]:
            if a in S and b in S:
                count += 1
        return count

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Choose a move
        # 1. Add a random element not in S
        # 2. Remove a random element in S
        # 3. Swap: remove one, add one
        
        move_type = random.random()
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates: continue
            x = random.choice(candidates)
            delta_E = count_conflicts(x, current_set)
            # New size = current_size + 1
            # New E = current_E + delta_E
            # Score change = 1 - 1000 * delta_E
            delta_score = 1 - 1000 * delta_E
            if delta_score > 0 or random.random() < pow(2.71828, -delta_score / T):
                current_set.add(x)
                current_E += delta_E
                current_size += 1
        elif move_type < 0.8:
            # Remove
            if not current_set: continue
            x = random.choice(list(current_set))
            delta_E = -count_conflicts(x, current_set)
            # New size = current_size - 1
            # New E = current_E + delta_E
            # Score change = -1 - 1000 * delta_E
            delta_score = -1 - 1000 * delta_E
            if delta_score > 0 or random.random() < pow(2.71828, -delta_score / T):
                current_set.remove(x)
                current_E += delta_E
                current_size -= 1
        else:
            # Swap
            if not current_set: continue
            candidates_out = list(current_set)
            candidates_in = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates_in: continue
            x_out = random.choice(candidates_out)
            x_in = random.choice(candidates_in)
            
            # Calculate delta_E for removing x_out and adding x_in
            # Note: The conflict count for x_in might change if x_out was part of a conflict with x_in?
            # No, x_in is not in S, so its conflicts are with pairs in S.
            # When we remove x_out, S changes.
            # So we should calculate:
            # 1. Remove x_out: E_new1 = E - count_conflicts(x_out, S)
            # 2. Add x_in to S\{x_out}: E_new2 = E_new1 + count_conflicts(x_in, S\{x_out})
            
            c_out = count_conflicts(x_out, current_set)
            # Temporarily remove x_out to check x_in?
            # Or just calculate count_conflicts(x_in, current_set) and subtract conflicts involving x_out?
            # count_conflicts(x_in, current_set) counts pairs (a,b) in current_set.
            # Some of these pairs might involve x_out.
            # If a pair (a,b) in bad_pairs_with[x_in] involves x_out, then after removing x_out, that pair is no longer in S.
            # So we need to be careful.
            
            # Let's just do it in two steps for correctness, even if slightly slower.
            # Or, we can just try the swap and recalculate E if accepted?
            # No, we need delta for SA.
            
            # Let's approximate:
            # c_in_full = count_conflicts(x_in, current_set)
            # c_in_partial = count_conflicts(x_in, current_set - {x_out})
            # This requires iterating bad_pairs_with[x_in] again.
            
            # Given the complexity, let's stick to Add/Remove for now, or implement Swap carefully.
            
            # Let's implement Swap carefully.
            c_out = count_conflicts(x_out, current_set)
            
            # To get c_in_partial, we iterate bad_pairs_with[x_in] and check if a,b are in current_set AND not x_out.
            c_in_partial = 0
            for a, b in bad_pairs_with[x_in]:
                if a == x_out or b == x_out:
                    continue
                if a in current_set and b in current_set:
                    c_in_partial += 1
            
            delta_E = -c_out + c_in_partial
            # Size change is 0.
            # Score change = -1000 * delta_E
            delta_score = -1000 * delta_E
            if delta_score > 0 or random.random() < pow(2.71828, -delta_score / T):
                current_set.remove(x_out)
                current_set.add(x_in)
                current_E += delta_E
                # current_size unchanged
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final check: Ensure validity. If not valid, remove elements until valid.
    # This is a safety net.
    # Since SA might end in an invalid state, we need to clean it up.
    # We can greedily remove elements that are part of conflicts.
    
    # Convert to list for final return
    final_list = list(current_set)
    
    # Verify validity
    def is_valid_set(s):
        s_set = set(s)
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                for k in range(j+1, len(s)):
                    a, b, c = s[i], s[j], s[k]
                    if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                        return False
        return True
        
    if not is_valid_set(final_list):
        # Remove elements to make it valid
        # Simple heuristic: remove the element involved in the most conflicts
        s_set = set(final_list)
        changed = True
        while changed:
            changed = False
            # Find a bad triangle
            bad_elem = None
            for i in range(len(final_list)):
                for j in range(i+1, len(final_list)):
                    for k in range(j+1, len(final_list)):
                        a, b, c = final_list[i], final_list[j], final_list[k]
                        if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                            bad_elem = a # Just pick one
                            changed = True
                            break
                    if changed: break
                if changed: break
            if bad_elem is not None:
                final_list.remove(bad_elem)
                s_set.remove(bad_elem)
                
    return final_list
# EVOLVE-BLOCK-END
