import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin before 90s limit
    
    # Precompute all 4-APs
    # For each element i (0-indexed), store the list of indices of the other 3 elements in each 4-AP it's part of
    # Actually, to make conflict counting fast, we need to know, for a given set S, how many 4-APs are fully contained in S.
    # Let's precompute all 4-APs as tuples of 4 indices (0-indexed).
    aps = []
    for a in range(N - 3):
        # a, a+d, a+2d, a+3d < N
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    # To speed up conflict checking, for each element i, store the list of APs that contain i.
    # And for each AP, store the indices of the other 3 elements.
    # But iterating over APs is slow if there are many.
    # Number of 4-APs in [1, 512] is roughly sum_{d=1}^{170} (512 - 3d) ~ 170*512 - 3*170*171/2 ~ 87040 - 43515 ~ 43525.
    # This is manageable.
    
    # For faster conflict counting:
    # Let c[i] be the number of 4-APs that contain element i and have all other 3 elements in S.
    # If we add x to S, the new number of conflicts (4-APs fully in S) increases by the number of APs containing x where the other 3 are already in S.
    # Let's precompute for each x, a list of lists of the other 3 elements for each AP containing x.
    
    aps_by_element = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            # Store the other 3 elements
            others = [j for j in ap if j != i]
            aps_by_element[i].append(others)
    
    # Initial set: Random or Greedy
    # Let's start with a greedy set to have a good initial state.
    # Greedy: Add elements in random order if they don't create a conflict.
    # To check if adding x creates a conflict, we check if any AP containing x has its other 3 elements in S.
    
    def greedy_init():
        S = np.zeros(N, dtype=bool)
        order = list(range(N))
        random.shuffle(order)
        for x in order:
            # Check if x can be added
            can_add = True
            for others in aps_by_element[x]:
                if S[others[0]] and S[others[1]] and S[others[2]]:
                    can_add = False
                    break
            if can_add:
                S[x] = True
        return S
    
    def count_conflicts(S):
        count = 0
        for ap in aps:
            if S[ap[0]] and S[ap[1]] and S[ap[2]] and S[ap[3]]:
                count += 1
        return count

    # Simulated Annealing
    # State: S (boolean array)
    # Objective: Maximize |S| subject to Conflicts(S) == 0.
    # We can use a penalty function: Score = |S| - M * Conflicts(S), where M is large.
    # Or just enforce Conflicts(S) == 0 and try to maximize |S|.
    # If we allow temporary conflicts, we can escape local optima.
    # Let's use Score = |S| - 100 * Conflicts(S).
    
    S = greedy_init()
    current_score = np.sum(S) - 100 * count_conflicts(S)
    best_S = S.copy()
    best_score = current_score
    
    T = 10.0
    cooling = 0.999
    
    # Precompute indices for faster access?
    # The main cost is checking conflicts when adding/removing.
    # When adding x, conflicts increase by: number of APs containing x where other 3 are in S.
    # When removing x, conflicts decrease by: number of APs containing x where other 3 are in S.
    
    def delta_conflicts_add(x, S):
        d = 0
        for others in aps_by_element[x]:
            if S[others[0]] and S[others[1]] and S[others[2]]:
                d += 1
        return d

    def delta_conflicts_remove(x, S):
        # Note: S[x] is True here.
        d = 0
        for others in aps_by_element[x]:
            if S[others[0]] and S[others[1]] and S[others[2]]:
                d += 1
        return d

    while time.time() < deadline:
        # Move: Flip a random element
        x = random.randint(0, N - 1)
        
        if S[x]:
            # Remove x
            delta_c = delta_conflicts_remove(x, S)
            delta_score = -1 - 100 * (-delta_c)  # |S| decreases by 1, Conflicts decreases by delta_c
            # Wait, Score = |S| - 100 * C.
            # New |S| = |S| - 1. New C = C - delta_c.
            # New Score = (|S|-1) - 100*(C-delta_c) = Score - 1 + 100*delta_c.
            delta_score = -1 + 100 * delta_c
        else:
            # Add x
            delta_c = delta_conflicts_add(x, S)
            # New |S| = |S| + 1. New C = C + delta_c.
            # New Score = (|S|+1) - 100*(C+delta_c) = Score + 1 - 100*delta_c.
            delta_score = 1 - 100 * delta_c
        
        if delta_score > 0:
            S[x] = not S[x]
            current_score += delta_score
        else:
            # Accept with probability exp(delta_score / T)
            if random.random() < np.exp(delta_score / T):
                S[x] = not S[x]
                current_score += delta_score
        
        if current_score > best_score:
            best_score = current_score
            best_S = S.copy()
        
        T *= cooling
        if T < 0.01:
            T = 1.0 # Restart or keep low? Let's keep it low for exploitation.
            # Actually, let's just let it cool down to 0.
            # But if we want to explore more, we might restart.
            # For now, simple cooling.
            # To prevent T from becoming too small and freezing, we can set a floor or restart.
            # Let's restart with a perturbed best_S occasionally.
            if random.random() < 0.1:
                # Perturb best_S
                S = best_S.copy()
                # Flip a few random bits
                for _ in range(10):
                    x = random.randint(0, N - 1)
                    S[x] = not S[x]
                current_score = np.sum(S) - 100 * count_conflicts(S)
                T = 5.0 # Reheat

    # Final cleanup: Remove any conflicts from best_S?
    # The best_S found might have conflicts if the penalty wasn't high enough or we accepted bad moves.
    # We need to return a VALID set.
    # So we must ensure the final set has 0 conflicts.
    # Let's take best_S and greedily remove elements that are part of conflicts until no conflicts remain.
    # Or better: Run a local search on best_S to remove conflicts while minimizing loss in size.
    
    final_S = best_S.copy()
    
    # Remove conflicts
    # While there is a conflict, remove one element from it.
    # To maximize size, we should remove the element that is in the most conflicts?
    # Or just remove any.
    
    # Let's do a simple loop:
    # Find a conflict, remove one element from it.
    # Repeat until no conflicts.
    
    # To be efficient, we can check conflicts.
    # Since we might have introduced conflicts, let's just run a "repair" phase.
    
    # A simpler repair:
    # Iterate through elements. If an element is part of a conflict, remove it?
    # No, that's too aggressive.
    
    # Let's use the conflict count function to guide removal.
    # Actually, since we used a penalty, best_S *should* have 0 conflicts if the search converged well.
    # But to be safe:
    
    # Check if best_S is valid.
    if count_conflicts(best_S) > 0:
        # Repair
        S_rep = best_S.copy()
        # Greedy removal
        # Find all conflicts
        # Remove elements involved in conflicts until none left.
        # This might reduce size significantly.
        # Better: Local search to remove conflicts.
        
        # Let's just try to remove elements that are in conflicts.
        # We'll do this in a loop.
        for _ in range(100): # Max iterations
            # Find a conflict
            conflict_found = False
            for ap in aps:
                if S_rep[ap[0]] and S_rep[ap[1]] and S_rep[ap[2]] and S_rep[ap[3]]:
                    # Remove one element from this AP
                    # Choose the one with the highest degree in the conflict graph?
                    # For simplicity, remove the first one.
                    # Or remove the one that is in the most other conflicts?
                    # Let's just remove ap[0].
                    S_rep[ap[0]] = False
                    conflict_found = True
                    break
            if not conflict_found:
                break
        
        if count_conflicts(S_rep) == 0:
            final_S = S_rep
        else:
            # Fallback: Greedy construction from scratch to ensure validity
            final_S = greedy_init()
            # Ensure it's valid (it should be)
            if count_conflicts(final_S) > 0:
                # This shouldn't happen with greedy_init
                # Just return a trivial set
                final_S = np.zeros(N, dtype=bool)
                final_S[0] = True

    # Convert to list of 1-indexed integers
    result = [i + 1 for i in range(N) if final_S[i]]
    
    # Final verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False

    if has_4ap(result):
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
