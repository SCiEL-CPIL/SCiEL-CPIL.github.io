import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General fallback: Direct sum or random
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists (balls of radius 1)
    # Each word has 1 + n*(q-1) = 1 + 7*2 = 15 neighbors
    neighbors_list = []
    neighbors_set = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbors_set.append(set(nb))

    # Precompute conflict graph adjacency using bitsets
    # Two words conflict if their Hamming distance is <= 2
    # We want to find a Maximum Independent Set (MIS) in this conflict graph,
    # which is equivalent to finding the smallest set of words such that no two are within distance 2?
    # Wait. The problem is a COVERING code.
    # A covering code of radius 1 is a set C such that for every word w, there exists c in C with d(w,c) <= 1.
    # This is NOT the same as a code with minimum distance 3 (which would be an independent set in the distance <= 2 graph).
    # The inspiration mentions "MIS approach on a conflict graph (nodes at distance <= 2)".
    # Let's re-read the inspiration carefully.
    # "The directive suggests using a maximum independent set (MIS) approach on a conflict graph (nodes at distance <= 2) to find the minimal code."
    # This seems to be a confusion in the inspiration or a specific trick.
    # Usually, Minimum Covering Code is NP-hard.
    # However, let's look at the current program. It uses Greedy Set Cover + Local Search (Kick and Repair).
    # The inspiration 1 (score 1.0253) actually uses a similar Greedy + Local Search but with random restarts and tie-breaking.
    # The "MIS" comment in the inspiration header might be misleading or referring to a different aspect (e.g. if we wanted to maximize the distance, it would be MIS).
    # But for covering, we want to MINIMIZE the number of centers.
    # Let's stick to the proven effective strategy: Greedy Set Cover + Local Search with kicks.
    # The current program scores 1.0210.
    # Inspiration 1 scores 1.0253.
    # What did Inspiration 1 do differently?
    # 1. It ran multiple greedy constructions with random tie-breaking.
    # 2. It selected the best initial code.
    # 3. It performed local search.
    #
    # The current program does a single deterministic greedy (mostly) and then local search.
    # The "Kicks" in the current program are random removals.
    #
    # Let's combine:
    # 1. Multiple random greedy starts (like Insp 1).
    # 2. A more robust local search that includes:
    #    - Redundancy removal (pruning).
    #    - Kick-and-repair with varying kick sizes.
    #    - Maybe a "swap" move: remove one, add one?
    #
    # Let's refine the local search to be more aggressive.
    # The key to beating 1.0210 (which is approx 186/184.8? No, score is seed/value.
    # Seed is usually the first valid result or a baseline.
    # If best known is 186, and lower bound is 156.
    # Score = seed / value.
    # If seed is 186 (best known), and we find 186, score is 1.0.
    # If we find 185, score is 186/185 = 1.0054.
    # If we find 184, score is 186/184 = 1.0108.
    # Current score 1.0210 implies value = 186 / 1.0210 = 182.17? So it found a code of size 182?
    # Or maybe the seed is different.
    # "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If the seed is the "best known" 186, then finding 182 gives 186/182 = 1.0219.
    # So the current program likely found a code of size 182.
    # Inspiration 1 found 1.0253 => 186 / 1.0253 = 181.4 => size 181.
    #
    # So the goal is to find a code of size < 182. Ideally 181 or lower.
    #
    # Strategy:
    # 1. Generate multiple good initial codes via randomized greedy.
    # 2. Apply intensive local search to each.
    # 3. Keep the best.
    #
    # Improvements to Local Search:
    # - The current `try_kick` removes k words and repairs greedily.
    # - We can also try "swap": remove 1, add 1. This is a 1-1 move.
    # - We can try "2-1": remove 2, add 1.
    # - We can try "1-2": remove 1, add 2. (This increases size, usually bad unless it enables future reductions).
    #
    # Let's implement a more generic local search that tries:
    # - Pruning redundant words.
    # - Random kicks of size k (remove k, repair).
    # - Random swaps (remove k1, add k2).
    #
    # To speed up the repair, we can precompute something?
    # The repair is the bottleneck.
    #
    # Let's try to optimize the greedy repair.
    #
    # Also, let's use the bitset idea if possible?
    # For N=2187, a bitset is 2187 bits = 274 bytes.
    # We can represent the set of covered words as a bitset.
    # When we remove a codeword, we subtract its neighbor bitset.
    # When we add one, we add its neighbor bitset.
    # This makes coverage checks very fast (bitwise AND/OR).
    #
    # Let's rewrite the core using bitsets for coverage.
    
    # Create bitsets for neighbors
    # Python ints can be used as arbitrary precision bitsets.
    # Bit i is 1 if word i is covered.
    
    neighbor_bitsets = []
    for i in range(N):
        mask = 0
        for j in neighbors_list[i]:
            mask |= (1 << j)
        neighbor_bitsets.append(mask)
    
    # Helper to count bits
    def popcount(x):
        return x.bit_count()

    def greedy_construct_random(seed=None):
        if seed is not None:
            random.seed(seed)
        
        covered = 0
        code = []
        code_set = set()
        
        # First pick 0
        code.append(0)
        code_set.add(0)
        covered |= neighbor_bitsets[0]
        
        total_uncovered = N - popcount(covered)
        
        while total_uncovered > 0:
            best_c = -1
            best_gain = 0
            best_candidates = []
            
            # We need to find the codeword that covers the most *uncovered* words.
            # uncovered_mask = ~covered & ((1 << N) - 1)
            # But N=2187, so (1<<N) is huge. Python handles it, but it's slow to create.
            # Instead, we can iterate over candidates and check gain.
            # Gain(c) = popcount(neighbor_bitsets[c] & ~covered)
            
            # Optimization: Only consider candidates that are not in code.
            # Also, we can restrict candidates to those that have at least one uncovered neighbor.
            # But checking that is as expensive as calculating gain.
            
            # Let's just scan all c not in code.
            # To speed up, we can maintain a list of "active" candidates?
            # Or just rely on the fact that N is small (2187).
            
            # Let's try to vectorize the gain calculation?
            # Not easily with Python ints.
            
            # Let's just loop.
            for c in range(N):
                if c in code_set:
                    continue
                gain = popcount(neighbor_bitsets[c] & ~covered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
            
            if best_gain == 0:
                break # Should not happen
            
            # Pick randomly from best candidates
            if len(best_candidates) > 1:
                chosen = random.choice(best_candidates)
            else:
                chosen = best_c
                
            code.append(chosen)
            code_set.add(chosen)
            covered |= neighbor_bitsets[chosen]
            total_uncovered -= best_gain # Wait, best_gain is the number of *newly* covered words?
            # Yes, because gain = popcount(bits in c's neighbors that are NOT in covered).
            # So we subtract best_gain from total_uncovered.
            
        return code, code_set, covered

    def is_redundant_bitset(c, covered, code_set):
        # Check if removing c leaves all its neighbors covered by others.
        # This is true if (covered & neighbor_bitsets[c]) == neighbor_bitsets[c] is NOT sufficient,
        # because 'covered' includes c's own contribution.
        # We need to check if every neighbor of c is covered by SOME OTHER codeword.
        # Let others_covered = covered & ~neighbor_bitsets[c] ? No.
        # Let's compute the coverage without c.
        # This requires knowing the coverage of others.
        # If we maintain a count, it's easy. With bitsets, it's hard to subtract.
        # 
        # Alternative: 
        # A word w in neighbors(c) is covered by others if there exists c' in code_set \ {c} such that w in neighbors(c').
        # This is equivalent to: (covered_without_c) & neighbor_bitsets[c] == neighbor_bitsets[c].
        # How to get covered_without_c?
        # We can't easily do it with a single bitset 'covered'.
        # 
        # So for redundancy check, we might need to revert to a count array or a slower check.
        # Or, we can check: for each w in neighbors(c), is w covered by any c' != c?
        # This is O(degree(c) * |code|) which is 15 * 180 ~ 2700 ops. Acceptable for a few checks.
        # But we do it for all c in code (180 times). 180 * 2700 = 486,000 ops.
        # This might be too slow if done every iteration.
        
        # Let's stick to the count array for redundancy, or just use the kick method which handles it implicitly.
        # The kick method removes words and checks if any are uncovered.
        # If no words are uncovered, the removal was safe (and potentially redundant words were removed).
        # So we don't strictly need a separate redundancy check if kicks are frequent.
        
        pass

    def greedy_repair_bitset(uncovered_mask, code_set, covered):
        """
        uncovered_mask is a Python int bitset of uncovered words.
        Returns list of added codewords and the new covered bitset.
        """
        added = []
        temp_uncovered = uncovered_mask
        new_covered = covered
        
        while temp_uncovered:
            best_c = -1
            best_gain = 0
            best_candidates = []
            
            # Candidates are those that cover at least one uncovered word.
            # We can find them by looking at neighbors of uncovered words?
            # Or just scan all c not in code_set.
            
            for c in range(N):
                if c in code_set:
                    continue
                gain = popcount(neighbor_bitsets[c] & temp_uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)
            
            if best_gain == 0:
                break
            
            if len(best_candidates) > 1:
                chosen = random.choice(best_candidates)
            else:
                chosen = best_c
                
            added.append(chosen)
            code_set.add(chosen) # Temporarily add to avoid re-picking
            new_covered |= neighbor_bitsets[chosen]
            temp_uncovered &= ~neighbor_bitsets[chosen]
            
        return added, new_covered

    # Main Search Loop
    
    best_code_size = float('inf')
    best_code = None
    best_code_set = None
    best_covered = None
    
    # Phase 1: Generate several good initial codes
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
        code, code_set, covered = greedy_construct_random(seed=i)
        if len(code) < best_code_size:
            best_code_size = len(code)
            best_code = code
            best_code_set = code_set
            best_covered = covered
    
    if best_code is None:
        code, code_set, covered = greedy_construct_random(seed=0)
        best_code = code
        best_code_set = code_set
        best_covered = covered
        best_code_size = len(code)

    code = best_code
    code_set = best_code_set
    covered = best_covered
    
    # Phase 2: Local Search
    # We will use a simulated annealing-like approach or just random kicks.
    # Since we want to minimize size, we accept any move that reduces size.
    # We can also accept moves that keep size same to escape local minima, but that's risky.
    # Let's stick to size-reducing moves primarily.
    
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline:
        # 1. Try to remove random words (Kick)
        # We try kicks of size 1, 2, 3.
        # If we remove k words and repair adds < k words, we improve.
        
        improved = False
        kick_size = random.choice([1, 1, 1, 2, 2, 3])
        
        if len(code) >= kick_size:
            # Pick kick_size random words
            indices = random.sample(range(len(code)), kick_size)
            removed = [code[i] for i in indices]
            
            # Temporarily remove
            temp_covered = covered
            for c in removed:
                # We can't just subtract bitsets because of overlaps.
                # We need to recalculate coverage or use counts.
                # This is the problem with bitsets for removal.
                # 
                # Let's revert to using a count array for the local search phase.
                # It's safer and easier to handle removals.
                pass
            
            # Actually, let's switch back to count arrays for the local search.
            # The bitset was good for greedy construction (fast AND).
            # For local search with removals, counts are better.
            
            break # Exit this loop, switch strategy

    # Let's restart the local search logic with count arrays.
    # We already have the best code from Phase 1.
    
    # Re-initialize counts for the best code
    coverage = np.zeros(N, dtype=np.int32)
    for c in code:
        coverage[neighbors_list[c]] += 1
        
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_indices):
        """
        uncovered_indices: list of word indices that are uncovered.
        Returns list of added codewords.
        """
        if not uncovered_indices:
            return []
        
        added = []
        temp_uncovered = set(uncovered_indices)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            best_candidates = []
            
            # Candidates are neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbors_set[w])
            
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = len(neighbors_set[c] & temp_uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)
            
            if best_count == 0:
                break
            
            if len(best_candidates) > 1:
                chosen = random.choice(best_candidates)
            else:
                chosen = best_c
                
            added.append(chosen)
            code_set.add(chosen)
            temp_uncovered.difference_update(neighbors_set[chosen])
            
        return added

    def apply_code_change(removed_list, added_list):
        nonlocal code, code_set, coverage
        for c in removed_list:
            code.remove(c)
            code_set.remove(c)
            coverage[neighbors_list[c]] -= 1
        for c in added_list:
            if c not in code_set:
                code.append(c)
                code_set.add(c)
                coverage[neighbors_list[c]] += 1

    def try_kick(k):
        if len(code) < k:
            return False
            
        indices = random.sample(range(len(code)), k)
        removed = [code[i] for i in indices]
        
        temp_coverage = coverage.copy()
        for c in removed:
            temp_coverage[neighbors_list[c]] -= 1
            
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            apply_code_change(removed, [])
            return True
            
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = len(code) - len(removed) + len(added)
        
        if new_size < len(code):
            apply_code_change(removed, added)
            return True
        else:
            return False

    # Main Local Search Loop
    while time.time() < deadline:
        # 1. Exhaustive Removal of Redundant Words
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for c in list(code):
                if is_redundant(c):
                    apply_code_change([c], [])
                    changed = True
                    if time.time() > deadline:
                        break
        
        # 2. Kicks
        kick_success = False
        # Try 1-removal
        if try_kick(1):
            kick_success = True
        # Try 2-removal
        elif try_kick(2):
            kick_success = True
        # Try 3-removal
        elif try_kick(3):
            kick_success = True
            
        if kick_success:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Perturb: add a random word? No, that increases size.
                # Maybe restart the local search phase with a slightly different code?
                # Or just keep trying.
                pass

    result = [idx_to_tuple(c) for c in code]
    return result
