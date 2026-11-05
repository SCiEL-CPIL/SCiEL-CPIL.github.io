import time
import random
import sys
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific task: K_3(6,1)
    # The evaluator passes q, n, radius.
    # If not the specific case, fall back to a simple greedy or direct sum.
    if not (q == 3 and n == 6 and radius == 1):
        # Fallback: Direct sum of tetracode if q=3, n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy for other cases (slow, but safe)
            all_words = list(itertools.product(range(q), repeat=n))
            code = []
            covered = set()
            for w in all_words:
                if w not in covered:
                    code.append(list(w))
                    for i in range(n):
                        for v in range(q):
                            if v == w[i]: continue
                            new_w = list(w)
                            new_w[i] = v
                            covered.add(tuple(new_w))
                    covered.add(w)
            return code

    # Specific case: q=3, n=6, radius=1
    start_time = time.time()
    deadline = 110.0 # 10s safety margin

    total = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute neighbors: neighbors[i] is a list of indices j such that dist(i,j) <= 1
    # There are 1 + n*(q-1) = 1 + 6*2 = 13 neighbors per word.
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        # Get digits
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i].append(i)
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i].append(new_i)
    
    # Convert to sets for fast membership testing during swap validity check
    neighbor_sets = [set(nei) for nei in neighbors]

    # Phase 1: Greedy Set Cover to get an initial code
    # We want to cover all 729 words.
    # Greedy: pick the word that covers the most currently uncovered words.
    uncovered = np.ones(total, dtype=np.int8)
    code_indices = set()
    counts = np.zeros(total, dtype=np.int32) # counts[u] = number of codewords covering u

    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        
        # Find candidate that covers most uncovered words
        # For each candidate c, score = sum(uncovered[neighbors[c]])
        # We can vectorize this if we treat neighbors as a matrix, but lists are fine for 729.
        best_candidate = -1
        best_score = 0
        
        # Optimization: Only consider candidates that are not already in code?
        # Actually, greedy set cover usually adds new candidates.
        # We can iterate over all 729 words.
        for c in range(total):
            # If c is already in code, it doesn't add new coverage? 
            # No, it's redundant. We only add if it covers *new* uncovered words.
            # Score is the number of uncovered words in N(c).
            # sum(uncovered[neighbors[c]])
            # Since neighbors[c] is small (13), this is fast.
            score = 0
            for u in neighbors[c]:
                if uncovered[u]:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        if best_score == 0:
            break # Should not happen if uncovered.sum() > 0 and code is valid cover? 
            # Actually if uncovered.sum() > 0, there must be a candidate with score > 0.
            # Unless we are stuck? No, greedy always finds one if uncovered exists.
            
        # Add best_candidate
        code_indices.add(best_candidate)
        for u in neighbors[best_candidate]:
            if uncovered[u]:
                uncovered[u] = 0
                counts[u] += 1
            else:
                # It was already covered, but we add another codeword covering it
                counts[u] += 1
        # Wait, the logic above for counts is slightly off.
        # counts[u] should be the total number of codewords in code_indices that cover u.
        # When we add best_candidate, we increment counts[u] for all u in N(best_candidate).
        # And we mark uncovered[u] = 0 for those u.
        
        # Let's re-verify the counts update.
        # If u was uncovered (1), it becomes covered (0). counts[u] increases by 1.
        # If u was covered (0), it remains covered (0). counts[u] increases by 1.
        # So simply:
        for u in neighbors[best_candidate]:
            counts[u] += 1
            uncovered[u] = 0

    # Check if we have a valid cover
    if uncovered.sum() != 0:
        # Fallback to direct sum if greedy failed or timed out badly
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_size = len(code_indices)
    best_code = list(code_indices)

    # Helper to check if a codeword can be removed
    def can_remove(c_idx):
        # All neighbors must have count > 1
        for nb in neighbors[c_idx]:
            if counts[nb] <= 1:
                return False
        return True

    # Helper to get the "removability score" of the current state
    # Max over all c in code of min(counts[nb] for nb in neighbors[c])
    def get_max_removability_score():
        max_score = 0
        for c in code_indices:
            min_count = min(counts[nb] for nb in neighbors[c])
            if min_count > max_score:
                max_score = min_count
        return max_score

    # Local Search Loop
    T = 1.0 # Temperature for simulated annealing
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        removed_any = False
        # Iterate over a copy
        for c in list(code_indices):
            if can_remove(c):
                # Remove c
                for nb in neighbors[c]:
                    counts[nb] -= 1
                code_indices.remove(c)
                removed_any = True
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
                break # Restart removal loop
        
        if removed_any:
            continue # Go back to top, try removing more
        
        # Phase 2: Swap with Simulated Annealing
        current_score = get_max_removability_score()
        
        # If current_score >= 2, we would have removed in Phase 1.
        # So current_score is likely 1 (or 0 if something is wrong).
        
        # Find a swap
        # We want to maximize the potential for removal.
        # Heuristic: Find a swap that increases the max_removability_score.
        # Or, if score stays 1, try to increase the "slack" of the bottleneck.
        
        # To avoid O(N^2) scan every time, we can be smart.
        # But N is ~75. 75*729 is ~55k checks. Each check is O(13). ~700k ops.
        # This is fast enough to do many iterations.
        
        best_swap = None
        best_swap_score = current_score
        best_swap_delta = 0
        
        # Sample c_out? No, check all c_out.
        # For c_in, we can't check all 729 for every c_out if we want speed, 
        # but 75 * 729 is small. Let's do it.
        
        # Optimization: Precompute which c_in cover which "critical" words?
        # Let's just iterate.
        
        for c_out in code_indices:
            # If c_out doesn't have any neighbor with count 1, it's not blocking removal directly?
            # Actually, if min(counts[N(c_out)]) > 1, it would be removed.
            # So every c_out has at least one neighbor with count 1.
            
            for c_in in range(total):
                if c_in in code_indices:
                    continue
                
                # Check validity of swap
                # Swap is valid if for all u in N(c_out) \ N(c_in), counts[u] >= 2
                # Because if u is in N(c_out) and NOT in N(c_in), its count decreases by 1.
                # If counts[u] was 1, it becomes 0 (invalid).
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Calculate new score
                # We don't want to apply the swap just to check score.
                # We can simulate the score change.
                # New counts:
                # counts[u] - 1 if u in N(c_out)
                # counts[u] + 1 if u in N(c_in)
                
                # The score is max_{c in C'} min_{u in N(c)} new_counts[u]
                # C' = (C \ {c_out}) U {c_in}
                
                # This is expensive to calculate exactly for every swap.
                # Approximation:
                # Focus on c_out and c_in.
                # The bottleneck words are those with count 1.
                # If the swap increases the count of any word u that is a neighbor of some c, 
                # and u was count 1, it might help.
                
                # Let's just apply the swap, check score, and revert if not accepted.
                # Applying swap is O(13+13) = 26 ops.
                # Checking score is O(|C| * 13) = 75 * 13 = 975 ops.
                # Total ~1000 ops per swap candidate.
                # 55k candidates * 1000 = 55M ops.
                # This might be too slow in Python for a single sweep.
                # We need to be smarter or limit candidates.
                
                # Strategy:
                # 1. Only consider c_in that cover at least one "critical" word (count 1) 
                #    that is NOT covered by c_out? 
                #    Actually, c_in should cover words that are "dangerously" low.
                
                # Let's limit c_in candidates to those that are "good".
                # A good c_in covers many words with count 1.
                # Let's precompute for each c_in, the number of neighbors with count 1.
                # But counts change.
                
                # Alternative:
                # Just pick a random valid swap?
                # Simulated Annealing works best with random moves.
                # Let's pick 10 random c_in for each c_out.
                
                pass # We will implement a restricted search below

        # Restricted Search:
        # For each c_out, pick a few promising c_in.
        # Promising c_in: covers many words with count 1.
        # Let's identify "critical words" (count 1).
        critical_words = [u for u in range(total) if counts[u] == 1]
        
        if not critical_words:
            # Should not happen if not removable? 
            # If no critical words, min count >= 2 for all words?
            # Then all codewords are removable.
            # This case is handled by Phase 1.
            break

        # For each c_out, find c_in that cover the most critical words.
        # We can't check all 729 c_in for each c_out.
        # Let's sample 20 c_in for each c_out.
        
        found_improvement = False
        for c_out in code_indices:
            # Sample 20 random c_in not in code
            candidates = random.sample([x for x in range(total) if x not in code_indices], min(20, len(code_indices) + total - len(code_indices)))
            
            for c_in in candidates:
                # Check validity
                valid = True
                for u in neighbors[c_out]:
                    if u not in neighbor_sets[c_in]:
                        if counts[u] < 2:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Apply swap temporarily
                for nb in neighbors[c_out]:
                    counts[nb] -= 1
                for nb in neighbors[c_in]:
                    counts[nb] += 1
                code_indices.remove(c_out)
                code_indices.add(c_in)
                
                new_score = get_max_removability_score()
                
                # Acceptance criteria
                accept = False
                if new_score > current_score:
                    accept = True
                else:
                    # Simulated Annealing
                    delta = current_score - new_score
                    if delta == 0:
                        accept = random.random() < 0.5 # Neutral move
                    else:
                        # delta > 0
                        accept_prob = np.exp(-delta / T)
                        accept = random.random() < accept_prob
                
                if accept:
                    found_improvement = True
                    # Keep the swap
                    # Try immediate removal
                    for c in list(code_indices):
                        if can_remove(c):
                            for nb in neighbors[c]:
                                counts[nb] -= 1
                            code_indices.remove(c)
                            if len(code_indices) < best_size:
                                best_size = len(code_indices)
                                best_code = list(code_indices)
                            break
                    break # Break out of c_in loop
                else:
                    # Revert swap
                    for nb in neighbors[c_in]:
                        counts[nb] -= 1
                    for nb in neighbors[c_out]:
                        counts[nb] += 1
                    code_indices.remove(c_in)
                    code_indices.add(c_out)
            
            if found_improvement:
                break # Break out of c_out loop
                
        if not found_improvement:
            # No swap accepted. Cool down.
            T *= cooling_rate
            if T < 0.01:
                # Restart with higher T? Or break?
                # Let's try to restart the local search with a perturbation.
                # Pick a random valid swap and apply it regardless of score?
                # Or just break.
                # Let's try one more time with T=1.0
                T = 1.0
                # Perturb: apply a random valid swap
                perturbed = False
                for _ in range(10):
                    c_out = random.choice(list(code_indices))
                    c_in = random.randint(0, total - 1)
                    if c_in in code_indices:
                        continue
                    valid = True
                    for u in neighbors[c_out]:
                        if u not in neighbor_sets[c_in]:
                            if counts[u] < 2:
                                valid = False
                                break
                    if valid:
                        for nb in neighbors[c_out]:
                            counts[nb] -= 1
                        for nb in neighbors[c_in]:
                            counts[nb] += 1
                        code_indices.remove(c_out)
                        code_indices.add(c_in)
                        perturbed = True
                        break
                if not perturbed:
                    break # Stuck

    # Convert best_code indices back to words
    return [list((best_code[i] // powers[p]) % q for p in range(n)) for i in range(len(best_code))]
