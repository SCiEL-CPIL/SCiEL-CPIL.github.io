import numpy as np
import random
import itertools
from collections import defaultdict

def construct(q, n, radius):
        # Hardcoded for q=3, n=7, radius=1 as per task description
        # But should be generic if possible? The task says "evaluator passes this task's q, n, radius".
        # So I should write it to handle the specific case efficiently, but the function signature is fixed.
        # The task specifically mentions "Football pool problem for 7 matches... ternary covering code K_3(7,1)".
        # It says "evaluator passes this task's alphabet size q, length n and covering radius".
        # So I must assume q=3, n=7, radius=1, but write code that takes args.
        
        if q != 3 or n != 7 or radius != 1:
             # Fallback to something reasonable if parameters change?
             # The task contract implies this specific instance.
             # I will implement the greedy for the general case but optimized for small N.
             pass

        # 1. Generate all words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # 2. Precompute neighbors (distance <= radius)
        # For radius 1, neighbors are self + 1-position changes.
        neighbors = [[] for _ in range(N)]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_word = list(w)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    neighbors[i].append(word_to_idx[new_word])
        
        # 3. Greedy Solver
        def solve_greedy(random_seed):
            random.seed(random_seed)
            uncovered = set(range(N))
            solution = []
            
            while uncovered:
                best_c = -1
                best_gain = -1
                # To introduce randomness in tie-breaking, we can shuffle indices or just pick random among max
                # Simple approach: iterate all, track max, collect all max, pick random from max.
                candidates_with_max = []
                for c in range(N):
                    # Count uncovered neighbors
                    gain = 0
                    for nb in neighbors[c]:
                        if nb in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates_with_max = [c]
                    elif gain == best_gain:
                        candidates_with_max.append(c)
                
                # Pick random from candidates_with_max
                if not candidates_with_max:
                    break # Should not happen
                best_c = random.choice(candidates_with_max)
                solution.append(best_c)
                
                # Update uncovered
                for nb in neighbors[best_c]:
                    if nb in uncovered:
                        uncovered.remove(nb)
            
            # Pruning Phase
            # Try to remove any codeword that is redundant
            # We check if the union of spheres of (solution \ {c}) covers all words.
            # Efficient check:
            # Maintain a count of how many codewords cover each word?
            # Let's just re-verify coverage for each candidate removal.
            # N_solution is small (~180). N_universe is 2187.
            # For each c in solution:
            #   Check if all words are covered by solution without c.
            #   This is expensive if done naively.
            
            # Better pruning:
            # Calculate `cover_count[word]` = number of codewords in solution covering `word`.
            # A codeword `c` is redundant if for all `w` in `neighbors[c]`, `cover_count[w] > 1`.
            # If we remove `c`, we decrement `cover_count` for its neighbors.
            # If any `cover_count` becomes 0, `c` is NOT redundant.
            
            cover_count = np.zeros(N, dtype=int)
            for c in solution:
                for nb in neighbors[c]:
                    cover_count[nb] += 1
            
            final_solution = list(solution)
            # Iterate through solution, try to remove
            # Order matters? Maybe random order.
            for c in final_solution[:]: # Copy to iterate safely
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    final_solution.remove(c)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
            
            return final_solution

        # Run multiple trials
        best_solution = None
        best_size = N
        
        # Number of trials depends on time limit (120s).
        # One trial takes ~0.1s? Let's assume 100 trials.
        for seed in range(100):
            sol = solve_greedy(seed)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
                # Early stopping if we hit known best?
                # Known best is 186. If we hit 186, we can stop?
                # Or try to go lower? Lower bound is 156.
                # Let's run a few more if we hit 186.
        
        # Convert indices back to words
        return [list(all_words[i]) for i in best_solution]
