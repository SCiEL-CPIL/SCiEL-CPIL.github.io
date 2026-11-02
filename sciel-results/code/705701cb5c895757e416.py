import time
import random
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # But we should handle general q, n, radius if possible, 
    # though the task says "evaluator passes this task's alphabet size...".
    # The seed program hardcodes logic for q=3, n=7. 
    # I will write a generic solver but optimized for the small q^n case.
    
    start_time = time.time()
    deadline = 100.0 # 20s buffer
    
    # Total words in space
    total_words = q ** n
    
    # If space is too large, this brute force won't work. 
    # But for 3^7 = 2187, it is very small.
    # If q^n is huge, we should fallback to a heuristic like the seed.
    if total_words > 100000:
        # Fallback to seed-like heuristic for large spaces
        # Direct sum or similar
        # For now, just return a trivial valid code if possible? 
        # Or just use the seed logic if it fits.
        # The task is specific to 3^7. 
        pass

    # Precompute neighbors for all words in {0..q-1}^n
    # Word is an integer 0..total_words-1
    # Convert int to base-q digits
    def int_to_tuple(x):
        return tuple(int(x // (q ** i) % q) for i in range(n))
    
    def tuple_to_int(t):
        val = 0
        for i, d in enumerate(t):
            val += d * (q ** i)
        return val

    # Precompute neighbors including self
    # neighbors[i] is a list of indices
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        t = int_to_tuple(i)
        # Self
        neighbors[i].append(i)
        # Radius 1 neighbors
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_new = list(t)
                t_new[pos] = new_val
                neighbors[i].append(tuple_to_int(tuple(t_new)))
        
    # Greedy Construction
    def greedy_construct():
        covered = [False] * total_words
        code = []
        # Count uncovered
        uncovered_count = total_words
        
        while uncovered_count > 0:
            best_gain = -1
            best_c = -1
            # Find best candidate
            for c in range(total_words):
                # Skip if c is already in code? 
                # Actually, adding a duplicate doesn't help. 
                # But we can just check gain. If gain is 0, skip.
                gain = 0
                for v in neighbors[c]:
                    if not covered[v]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_gain <= 0:
                break # Should not happen if uncovered_count > 0
            code.append(best_c)
            # Update covered
            for v in neighbors[best_c]:
                if not covered[v]:
                    covered[v] = True
                    uncovered_count -= 1
        return code

    # Initial Greedy
    code = greedy_construct()
    
    # Local Search / Improvement
    # Try to remove codewords
    # We represent coverage by a count array? 
    # Or just recompute coverage from scratch? 
    # Recomputing is O(|C| * 22). |C| ~ 186. 186*22 = 4000 ops. 
    # Checking validity is O(2187). 
    # So one check is ~6000 ops. Very fast.
    
    def is_covering(code):
        covered = [False] * total_words
        for c in code:
            for v in neighbors[c]:
                covered[v] = True
        # Check if all covered
        # sum(covered) == total_words
        return all(covered)

    def get_uncovered(code):
        covered = [False] * total_words
        for c in code:
            for v in neighbors[c]:
                covered[v] = True
        return [i for i in range(total_words) if not covered[i]]

    # Iterative removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try removing each codeword
        for i in range(len(code)):
            if (time.time() - start_time) > deadline:
                return code
            candidate_code = code[:i] + code[i+1:]
            if is_covering(candidate_code):
                code = candidate_code
                improved = True
                break # Restart loop as code changed
    
    # If we have time, try "Swap" heuristic to break local minima?
    # If removal fails, it means the removed word covered some unique words.
    # Maybe a different word could cover those unique words AND allow another word to be removed?
    # This is complex. Given 186 is the target and greedy usually gets close, 
    # and the removal loop is strong, let's see if we need swaps.
    # For 3^7, greedy often gets 189-190. Removal might get to 186-188.
    
    # Let's add a randomization to greedy to find better starting points?
    # We can run multiple greedy instances with different tie-breaking.
    
    best_code = code
    best_len = len(code)
    
    # Run a few more greedy iterations with random tie-breaking if time permits
    # The greedy step is the bottleneck if we do it many times.
    # One greedy run is ~ 186 * 2187 * 22 ~ 9M ops. ~ 0.5s in Python? 
    # Let's check time.
    
    if (time.time() - start_time) < deadline - 10:
        for _ in range(5): # Try a few more
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Randomized Greedy
            covered = [False] * total_words
            code_rand = []
            uncovered_count = total_words
            
            while uncovered_count > 0:
                best_gain = -1
                candidates = []
                for c in range(total_words):
                    gain = 0
                    for v in neighbors[c]:
                        if not covered[v]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [c]
                    elif gain == best_gain:
                        candidates.append(c)
                
                if best_gain <= 0:
                    break
                
                # Pick random from best candidates
                best_c = random.choice(candidates)
                code_rand.append(best_c)
                for v in neighbors[best_c]:
                    if not covered[v]:
                        covered[v] = True
                        uncovered_count -= 1
            
            # Improve this random greedy result
            improved = True
            while improved and (time.time() - start_time) < deadline - 2:
                improved = False
                for i in range(len(code_rand)):
                    if (time.time() - start_time) > deadline - 2:
                        break
                    candidate_code = code_rand[:i] + code_rand[i+1:]
                    if is_covering(candidate_code):
                        code_rand = candidate_code
                        improved = True
                        break
                if (time.time() - start_time) > deadline - 2:
                    break
            
            if len(code_rand) < best_len:
                best_code = code_rand
                best_len = len(code_rand)
                
    return [list(int_to_tuple(c)) for c in best_code]

# EVOLVE-BLOCK-END
