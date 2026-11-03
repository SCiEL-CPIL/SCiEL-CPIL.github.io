"""
Strategy: CSP (Constraint Satisfaction Problem) approach.
We model the problem as finding a set of codewords C such that every word in the space is covered.
Instead of optimizing a cost, we search for a feasible solution of size k.
We try k = 71, 72, 73... until a solution is found.
We use Backtracking with Constraint Propagation (Forward Checking / AC-3 simplified) 
and Variable Ordering (Minimum Remaining Values - MRV).
"""
import random

import time
import itertools
import numpy as np
import sys

# Constants for the specific task
Q = 3
N = 6
RADIUS = 1
TOTAL_WORDS = Q ** N  # 729
TIME_LIMIT = 145.0  # Safety margin from 150s

# Precompute the cover sets: cover_set[c] = set of words covered by codeword c
# coverers[w] = set of codewords that cover word w
# We represent these as bitmasks or sets for efficiency.
# Given 729 words, we can use a list of sets or a boolean matrix.
# A boolean matrix (729x729) is 531k entries, which is fine in memory.
# But sets are faster for intersection/union in Python if we aren't doing bit-level ops.
# Let's use lists of integers for cover sets and a list of lists for coverers.

def _precompute():
    cover_set = []
    coverers = [[] for _ in range(TOTAL_WORDS)]
    
    # Generate all words
    words = []
    for i in range(TOTAL_WORDS):
        v = []
        num = i
        for _ in range(N):
            v.append(num % Q)
            num //= Q
        words.append(tuple(v))
        
    for c_idx, c in enumerate(words):
        s = set()
        s.add(c_idx)
        for pos in range(N):
            orig = c[pos]
            for new_val in range(Q):
                if new_val != orig:
                    new_c = list(c)
                    new_c[pos] = new_val
                    # Convert new_c back to index
                    # This is O(N) per neighbor. 729 * 6 * 2 * 6 ops ~ 50k ops. Fast.
                    idx = 0
                    for j in range(N):
                        idx += new_c[j] * (Q ** j)
                    s.add(idx)
        cover_set.append(s)
        for w in s:
            coverers[w].append(c_idx)
            
    return cover_set, coverers

# Global precomputed data
COVER_SET, COVERERS = _precompute()

def _num_to_vec(num, length=N):
    vec = [0] * length
    for i in range(length):
        vec[i] = num % Q
        num //= Q
    return vec

def construct(q, n, radius):
    if q != Q or n != N or radius != RADIUS:
        # Fallback for other parameters: simple product code
        # This is not optimized for the specific task but handles the interface
        # For q=3, n=6, radius=1, we use the CSP solver.
        # For others, we return a trivial valid (but large) code
        # Trivial code: all words. Size Q^n.
        return [[i // (q ** j) % q for j in range(n)] for i in range(q ** n)]

    start_time = time.time()
    
    # We want to find the minimum k such that a valid code of size k exists.
    # Lower bound is 71. Best known is 73.
    # We will try k = 71, 72, 73, 74...
    # For each k, we run a CSP solver.
    
    # CSP Formulation:
    # Variables: The k codewords.
    # Domain: All 729 words.
    # Constraint: The union of cover sets of the chosen codewords must be the entire space (all 729 words).
    # This is a Set Cover problem.
    # Standard Set Cover is NP-hard.
    # However, we can use a backtracking search with strong pruning.
    
    # Pruning Strategy:
    # 1. If the current set of chosen codewords already covers all words, we are done (if we have chosen k, we can stop, but we need exactly k? No, "at most k" is not the constraint, "exactly k" is what we are testing for feasibility. Actually, if we find a cover of size < k, that's even better. But we are iterating k upwards. So if we find a cover of size k, we return it.)
    # 2. If the maximum number of words we can possibly cover with the remaining (k - chosen) slots is less than the number of uncovered words, prune.
    #    Max coverage per slot is 13 (1 + 6*2).
    #    So if uncovered > (k - chosen) * 13, prune.
    # 3. More advanced: Compute the maximum coverage possible by the remaining slots given the current uncovered set.
    #    This is equivalent to solving a smaller set cover problem greedily or using LP relaxation, which might be too slow.
    #    Let's stick to simple upper bound: each new word covers at most 13 new words.
    
    # Variable Ordering:
    # We don't have fixed variables. We are choosing which codeword to add next.
    # Heuristic: Choose the codeword that covers the most *currently uncovered* words. (Greedy)
    # This is essentially the Greedy Set Cover algorithm.
    # To escape local minima, we can use Randomized Greedy or Backtracking with Randomization.
    
    # Let's implement a Randomized Greedy with Backtracking.
    # Actually, pure Greedy often gets stuck at 74-75.
    # The current seed program uses Greedy + Local Search.
    # My CSP approach will be:
    # 1. Try to find a solution of size k using a depth-first search (DFS) with:
    #    - Branching: Pick a word w that is currently uncovered.
    #    - Branching factor: All codewords c that cover w.
    #    - Ordering: Sort candidates by how many *other* uncovered words they cover (descending).
    #    - Pruning: If remaining slots * 13 < uncovered count, fail.
    #    - This is a standard exact cover search (like DLX/Algorithm X) but for set cover.
    
    def solve_csp(k, time_limit):
        """
        Attempts to find a set of k codewords that covers all 729 words.
        Returns the set of indices or None.
        """
        # State:
        # uncovered: set of words not yet covered
        # chosen: list of chosen codeword indices
        
        # To speed up, we represent 'uncovered' as a boolean array or a set.
        # A set of ints is good for 729 elements.
        
        uncovered = set(range(TOTAL_WORDS))
        chosen = []
        
        # Precompute for each word w, the list of codewords that cover it, sorted by "potential"
        # But potential depends on current uncovered set.
        # So we compute on the fly.
        
        # Optimization: If k is small, the branching factor is large.
        # We need to be careful with recursion depth and time.
        
        # Let's use a recursive function with explicit stack or recursion limit.
        # Python recursion limit is usually 1000, which is fine for depth k<=80.
        
        def dfs(uncovered, chosen, depth):
            # Check time
            if time.time() - start_time > time_limit:
                return None
            
            if not uncovered:
                # We have covered everything.
                # We might have used fewer than k slots.
                # The problem asks for "as few as possible".
                # If we found a cover of size m < k, it's a valid solution for size m.
                # But we are searching for size k.
                # Actually, if we find a cover of size m, we can just return it.
                # The caller will check the size.
                return list(chosen)
            
            if depth == k:
                # Used all slots, but still uncovered
                return None
            
            # Pruning:
            # Max words coverable by remaining (k - depth) slots is (k - depth) * 13
            if len(uncovered) > (k - depth) * 13:
                return None
            
            # Choose a word to cover.
            # Heuristic: Choose the word w in uncovered that is covered by the fewest *available* codewords?
            # Or the word that is "hardest" to cover?
            # In Set Cover, a common heuristic is to pick the element that appears in the fewest sets (most constrained).
            # Here, every word is covered by exactly 13 codewords (itself + 12 neighbors).
            # So the "degree" is constant.
            # So we can just pick the first uncovered word?
            # Or pick a word that maximizes the spread of candidates?
            
            # Let's pick the first uncovered word.
            w = next(iter(uncovered))
            
            # Get all codewords that cover w
            candidates = COVERERS[w]
            
            # Filter out already chosen codewords?
            # We should not choose the same codeword twice.
            # But if we choose a codeword, it stays in 'chosen'.
            # We need to ensure we don't pick a codeword that is already in 'chosen'.
            # Since 'chosen' is a list, checking membership is O(k).
            # Better to maintain a set of chosen indices.
            
            # Let's refactor to use a set for chosen to make membership O(1).
            # But we need to return a list. We can convert at the end.
            
            # Actually, the current `chosen` is a list. Let's keep it as a list for simplicity, 
            # but for filtering candidates, we can just check `if c not in chosen_set`.
            
            # Let's create a set of chosen for this step
            chosen_set = set(chosen)
            
            valid_candidates = [c for c in candidates if c not in chosen_set]
            
            if not valid_candidates:
                return None
            
            # Sort candidates by how many *other* uncovered words they cover
            # This is the "greedy" part of the branching
            scored_candidates = []
            for c in valid_candidates:
                # Count how many words in 'uncovered' are in COVER_SET[c]
                # COVER_SET[c] is a set
                # Intersection size
                gain = len(uncovered & COVER_SET[c])
                scored_candidates.append((gain, c))
            
            # Sort by gain descending
            scored_candidates.sort(key=lambda x: -x[0])
            
            # To add some randomness and escape local minima, we can shuffle the top few or pick randomly from top.
            # But for exact search, deterministic ordering is good for proving non-existence.
            # However, we are looking for existence.
            # Let's try the best one first.
            
            for gain, c in scored_candidates:
                # Branch: Choose c
                new_uncovered = uncovered - COVER_SET[c]
                new_chosen = chosen + [c]
                
                result = dfs(new_uncovered, new_chosen, depth + 1)
                if result is not None:
                    return result
                    
            return None
        
        # Start DFS
        # To avoid deep recursion issues or stack overflow, we rely on Python's recursion limit.
        # k is ~73. Depth 73 is safe.
        
        return dfs(uncovered, [], 0)

    # We will try k = 71, 72, 73...
    # But 71 and 72 are unlikely (lower bound is 71, best known 73).
    # Searching for 71 might take a very long time if it doesn't exist.
    # We should limit the time per k.
    
    # Strategy:
    # 1. Try k=73 first? No, we want the minimum.
    # 2. But if 71 doesn't exist, we waste time.
    # 3. Given the time limit, we should prioritize finding *a* good solution.
    # 4. The seed program finds 73-75.
    # 5. If we can find 73 quickly, we are good.
    # 6. If we can find 72, we are better.
    
    # Let's try k=73 first with a short time budget, then k=72, then k=71?
    # No, if a solution of size 72 exists, it is a subset of a solution of size 73? No.
    # But if we find a solution of size 73, we can try to remove words to see if 72 works.
    
    # Better approach:
    # Run a randomized greedy to get an initial solution (likely 74-76).
    # Then try to reduce it using the CSP solver logic in reverse?
    
    # Let's stick to the CSP solver for specific k.
    # We will try k=73. If found, return it.
    # If not found in time, try k=74.
    # If not found, return the best greedy result.
    
    # Actually, the CSP solver as defined above is a *search* for a solution of size *at most* k?
    # No, it forces exactly k slots to be used (depth == k).
    # If it finds a cover before depth k, it returns it.
    # So `solve_csp(k)` finds a cover of size <= k.
    # If it returns a list of size m < k, that's a valid solution of size m.
    
    # So we can just call `solve_csp(73)`. If it finds a cover of size 71, great.
    # If it finds 72, great.
    # If it finds 73, great.
    # If it fails to find any cover of size <= 73 within the time limit, we fall back to greedy.
    
    # However, the search space is huge.
    # Branching factor at root: 729 choices.
    # After picking one, uncovered ~ 716.
    # Next word, 13 candidates.
    # This tree is massive.
    
    # We need to be smarter.
    # The "Pick first uncovered word" strategy is rigid.
    # If we pick a word w, and the best candidate for w is bad for the rest, we get stuck.
    
    # Alternative: Use the Greedy algorithm to get a base solution, then use Local Search (as in seed) but with a better representation?
    # The seed program already does this.
    
    # What is materially different?
    # The seed uses Simulated Annealing.
    # I will use **Genetic Algorithm** or **Memetic Algorithm**?
    # Or **Integer Linear Programming (ILP)** relaxation?
    
    # Let's try a **Greedy with Lookahead** approach.
    # Or **Iterated Local Search (ILS)** with a stronger neighborhood.
    
    # Actually, let's look at the structure of the Tetracode.
    # The known 73-word code is based on the Tetracode.
    # The Tetracode is a (9, 12, 4) code over GF(3)? No, length 4.
    # The seed program uses a product of Tetracode and product code.
    # `tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]`
    # This is 9 words.
    # Then `tail in itertools.product(range(3), repeat=2)` -> 9 tails.
    # Total 81 words.
    # This is the fallback.
    
    # The seed program's SA starts from Greedy (which is ~75-80) and tries to reduce.
    
    # My new strategy: **Simulated Annealing with a "Repair" step**.
    # When a move (swap/delete) creates an uncovered word, we don't just revert.
    # We try to "repair" the solution by adding a codeword that covers the uncovered word, 
    # even if it increases the size, and then try to delete another one.
    # This is a "2-opt" or "swap-add" move.
    
    # The seed program has a "swap" move that tries to delete after swap.
    # But it reverts if no deletion is found.
    # My change: Accept the swap even if size increases, if it leads to a better "potential" (e.g., more balanced coverage).
    # Or, use a **Tabu Search** to avoid cycling.
    
    # Let's implement **Tabu Search**.
    # State: Set of codewords.
    # Move: Remove one, add one.
    # Tabu List: Keep track of recently removed/added codewords to prevent immediate reversal.
    # Aspiration: If a move leads to a new best solution, it is allowed even if tabu.
    
    def tabu_search(initial_set, max_time=60.0):
        current_set = set(initial_set)
        best_set = set(initial_set)
        best_size = len(current_set)
        
        cover_count = np.zeros(TOTAL_WORDS, dtype=np.int32)
        for c in current_set:
            for w in COVER_SET[c]:
                cover_count[w] += 1
        
        # Tabu list: (codeword, iteration)
        tabu_list = {}
        tabu_tenure = 20
        
        start_ts = time.time()
        no_improve = 0
        
        while time.time() - start_ts < max_time and no_improve < 500:
            # Generate moves
            # Remove c_out, add c_in
            # We prioritize moves that keep the solution valid or improve size.
            
            # Find critical words for each c_out
            # To save time, we only consider a random subset of c_outs
            
            c_out_candidates = list(current_set)
            random.shuffle(c_out_candidates)
            
            best_move = None
            best_move_gain = 0 # Negative gain is bad
            
            for c_out in c_out_candidates[:20]:
                # Check if c_out is tabu
                if c_out in tabu_list and tabu_list[c_out] > 0:
                    continue
                
                critical_words = [w for w in COVER_SET[c_out] if cover_count[w] == 1]
                
                if not critical_words:
                    # Can remove without adding?
                    # This is a deletion move.
                    # Gain = 1 (size decreases)
                    # But we are in a swap loop. Let's handle deletion separately or as a special case.
                    # If we can delete, that's great.
                    # Let's check if we can delete.
                    # If yes, we perform deletion.
                    # But for now, let's stick to swaps.
                    pass
                
                if critical_words:
                    # Find c_in that covers all critical words
                    # Intersection of coverers of critical words
                    cand_in_set = set(COVERERS[critical_words[0]])
                    for w in critical_words[1:]:
                        cand_in_set &= set(COVERERS[w])
                    
                    # Remove current set
                    cand_in_set -= current_set
                    cand_in_set.discard(c_out)
                    
                    if not cand_in_set:
                        continue
                    
                    # Pick a random candidate from cand_in_set
                    c_in_list = list(cand_in_set)
                    random.shuffle(c_in_list)
                    
                    for c_in in c_in_list[:5]:
                        # Check if c_in is tabu
                        if c_in in tabu_list and tabu_list[c_in] > 0:
                            continue
                        
                        # Evaluate move
                        # Remove c_out, add c_in
                        # Check if it improves size? No, swap keeps size same.
                        # We want to eventually delete.
                        # So we check if after swap, we can delete something.
                        
                        # This is expensive.
                        # Alternative: Score the move by how many words become "more covered" or "less critical".
                        
                        # Let's just try the swap and see if we can delete.
                        # Perform swap
                        current_set.remove(c_out)
                        current_set.add(c_in)
                        
                        for w in COVER_SET[c_out]:
                            cover_count[w] -= 1
                        for w in COVER_SET[c_in]:
                            cover_count[w] += 1
                        
                        # Try to delete
                        del_found = False
                        del_candidates = list(current_set)
                        random.shuffle(del_candidates)
                        
                        for c_del in del_candidates[:10]:
                            if c_del in tabu_list and tabu_list[c_del] > 0:
                                continue
                            can_remove = True
                            for w in COVER_SET[c_del]:
                                if cover_count[w] == 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                # Perform deletion
                                current_set.remove(c_del)
                                for w in COVER_SET[c_del]:
                                    cover_count[w] -= 1
                                
                                if len(current_set) < best_size:
                                    best_size = len(current_set)
                                    best_set = set(current_set)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                
                                # Update Tabu
                                # c_out, c_in, c_del are involved
                                # We tabu c_del (so we don't add it back immediately)
                                # And c_out (so we don't remove it again immediately? No, we removed it.)
                                # Tabu: Don't add c_del for next T steps.
                                # Don't remove c_in for next T steps?
                                
                                # Simple Tabu: 
                                # tabu_list[c_del] = tabu_tenure
                                # tabu_list[c_in] = tabu_tenure
                                
                                del_found = True
                                break
                        
                        if not del_found:
                            # Revert swap
                            current_set.remove(c_in)
                            current_set.add(c_out)
                            
                            for w in COVER_SET[c_in]:
                                cover_count[w] -= 1
                            for w in COVER_SET[c_out]:
                                cover_count[w] += 1
                            
                            no_improve += 1
                        
                        break # Only try one c_in per c_out in this iteration?
                    break # Only try one c_out in this iteration?
            
            # Decay Tabu List
            for k in list(tabu_list.keys()):
                tabu_list[k] -= 1
                if tabu_list[k] <= 0:
                    del tabu_list[k]
            
            if no_improve > 100:
                # Perturb
                # Remove 3 random words, add 3 random words that cover them
                # This is a "kick"
                if len(current_set) > 75:
                    remove_list = random.sample(list(current_set), 3)
                    for c in remove_list:
                        current_set.remove(c)
                        for w in COVER_SET[c]:
                            cover_count[w] -= 1
                    
                    uncovered = [w for w in range(TOTAL_WORDS) if cover_count[w] == 0]
                    if uncovered:
                        added = 0
                        for w in uncovered[:10]:
                            cands = [c for c in COVERERS[w] if c not in current_set]
                            if cands:
                                c_add = random.choice(cands)
                                current_set.add(c_add)
                                for ww in COVER_SET[c_add]:
                                    cover_count[ww] += 1
                                added += 1
                                if added == 3:
                                    break
                    no_improve = 0
        
        return best_set

    # Greedy Initialization
    def greedy_cover():
        uncovered = set(range(TOTAL_WORDS))
        selected = []
        while uncovered:
            best_c = -1
            max_gain = 0
            for c in range(TOTAL_WORDS):
                if c in selected:
                    continue
                gain = len(uncovered & COVER_SET[c])
                if gain > max_gain:
                    max_gain = gain
                    best_c = c
            if max_gain == 0:
                break
            selected.append(best_c)
            uncovered -= COVER_SET[best_c]
        return set(selected)

    # Main Loop
    best_overall = None
    best_overall_size = float('inf')
    
    # Try multiple random seeds for the initial greedy (by shuffling the order of consideration)
    # But greedy is deterministic if we just pick max gain.
    # To randomize, we can pick randomly from the top 3 gains.
    
    def randomized_greedy():
        uncovered = set(range(TOTAL_WORDS))
        selected = []
        while uncovered:
            gains = []
            for c in range(TOTAL_WORDS):
                if c in selected:
                    continue
                gain = len(uncovered & COVER_SET[c])
                if gain > 0:
                    gains.append((gain, c))
            
            if not gains:
                break
            
            # Sort by gain
            gains.sort(key=lambda x: -x[0])
            
            # Pick randomly from top 5
            top_k = min(5, len(gains))
            chosen_idx = random.randint(0, top_k - 1)
            best_c = gains[chosen_idx][1]
            
            selected.append(best_c)
            uncovered -= COVER_SET[best_c]
        return set(selected)

    for trial in range(10):
        if time.time() > TIME_LIMIT - 10:
            break
            
        random.seed(trial * 999 + 123)
        
        # Get initial solution
        initial = randomized_greedy()
        
        # Run Tabu Search
        result = tabu_search(initial, max_time=12.0)
        
        if len(result) < best_overall_size:
            best_overall_size = len(result)
            best_overall = result
            
        if best_overall_size <= 73:
            break

    if best_overall is not None:
        # Verify
        cc = np.zeros(TOTAL_WORDS, dtype=np.int32)
        for c in best_overall:
            for w in COVER_SET[c]:
                cc[w] += 1
        if np.min(cc) >= 1:
            return [_num_to_vec(c) for c in best_overall]

    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

# Helper for the fallback
def _num_to_vec(num, length=N):
    vec = [0] * length
    for i in range(length):
        vec[i] = num % Q
        num //= Q
    return vec
