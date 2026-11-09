import time
import itertools
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    N = q ** n
    
    # Helper: tuple to index
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx

    # Helper: index to tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        t.reverse()
        return tuple(t)

    # Precompute all words
    all_words = [idx_to_tuple(i) for i in range(N)]
    
    # Precompute balls: for each center, list of indices covered
    # For q=3, n=7, radius=1: ball size is 15
    balls = []
    for i in range(N):
        w = list(all_words[i])
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                old_val = w[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        w[pos] = new_val
                        j = tuple_to_idx(tuple(w))
                        ball.append(j)
                w[pos] = old_val
        balls.append(ball)
        
    # Bitset representation for speed
    # Each ball is a bitset of N bits
    # N=2187, so we need 35 64-bit integers (2240 bits)
    num_words = (N + 63) // 64
    ball_bitsets = []
    for b in balls:
        bits = [0] * num_words
        for idx in b:
            bits[idx // 64] |= (1 << (idx % 64))
        ball_bitsets.append(bits)
        
    def bitset_union(bits_list):
        res = [0] * num_words
        for bits in bits_list:
            for i in range(num_words):
                res[i] |= bits[i]
        return res

    def bitset_and(bits1, bits2):
        res = [0] * num_words
        for i in range(num_words):
            res[i] = bits1[i] & bits2[i]
        return res

    def bitset_count(bits):
        return sum(bin(b).count('1') for b in bits)

    def bitset_subtract(bits1, bits2):
        # bits1 - bits2 (only where bits1 has 1 and bits2 has 0)
        res = [0] * num_words
        for i in range(num_words):
            res[i] = bits1[i] & ~bits2[i]
        return res

    def bitset_eq(bits1, bits2):
        for i in range(num_words):
            if bits1[i] != bits2[i]:
                return False
        return True

    def bitset_is_subset(bits1, bits2):
        # Is bits1 subset of bits2?
        for i in range(num_words):
            if bits1[i] & ~bits2[i]:
                return False
        return True

    # Greedy initialization
    code_indices = []
    covered_bits = [0] * num_words
    uncovered_count = N
    
    # Precompute gain for each candidate? No, dynamic.
    # Simple greedy: pick codeword that covers max new words
    # To speed up, we can just iterate until covered.
    
    # Optimization: Use a priority queue or just max scan.
    # N=2187 is small enough for O(N^2) if optimized, but let's do standard greedy.
    
    # Actually, a better greedy is:
    # 1. Start with empty code.
    # 2. While not covered:
    #    a. For each unselected i, calculate gain = count(ball_i \ covered)
    #    b. Pick max gain.
    #    c. Add to code, update covered.
    
    # Since N is small (2187), and ball size is small (15), we can represent
    # "covered" as a boolean array for fast update, but use bitsets for gain calculation?
    # No, boolean array is faster for updates. Bitsets are faster for set ops on large sets.
    # Let's stick to boolean array for the main loop, it's simpler and N is small.
    
    covered_bool = [False] * N
    code = []
    
    while any(not c for c in covered_bool):
        best_gain = -1
        best_idx = -1
        # Heuristic: prioritize indices that are not yet covered?
        # Standard greedy:
        for i in range(N):
            if i in code: # Skip already selected (optimization: keep a set)
                continue
            gain = 0
            for j in balls[i]:
                if not covered_bool[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            # Early exit if gain is 0 (shouldn't happen if not covered)
            if gain == 0:
                break
        if best_gain == 0:
            break
        code.append(best_idx)
        for j in balls[best_idx]:
            covered_bool[j] = True
            
    # Convert code to bitset
    current_code_bits = [0] * num_words
    for i in code:
        for b_idx, b_val in enumerate(ball_bitsets[i]):
            current_code_bits[b_idx] |= b_val
            
    # Local Search: 1-swap and 1-delete
    # We maintain a set of selected indices for fast membership check
    selected_set = set(code)
    
    def try_remove(i):
        nonlocal current_code_bits, selected_set, code
        # Check if removing i leaves any word uncovered
        # Union of other balls must cover ball_i
        # Specifically, for each j in ball_i, there must be some k in code\{i} such that j in ball_k
        # This is expensive.
        # Alternative: 
        # new_covered = current_covered \ ball_i (conceptually)
        # We need to check if (current_covered \ ball_i) is still full.
        # i.e., is there any j in ball_i such that NO other codeword covers j?
        
        # Precompute for each j, how many codewords cover it?
        # This requires maintaining a count array.
        pass

    # Better approach for local search with bitsets:
    # Maintain a "coverage count" array? No, N=2187 is small.
    # Let's maintain a list of "uncovered" indices? No, we want to minimize size.
    
    # Let's use a different metric for local search:
    # Objective: Minimize |code| subject to Union(ball_i) = All.
    
    # 1-Delete:
    # For each i in code:
    #   Check if Union(ball_k for k in code\{i}) == All.
    #   This is equivalent to: Is there any j in ball_i such that j is NOT in Union(ball_k for k in code\{i})?
    #   Let U_except_i = Union(ball_k for k in code\{i})
    #   We need U_except_i to cover ball_i.
    #   i.e., ball_i \ U_except_i must be empty.
    
    # To compute U_except_i efficiently:
    # We can precompute the union of all balls in code.
    # But removing one ball from the union is not trivial with bitsets (no subtraction of union).
    
    # Alternative:
    # For each i in code, check if i is "redundant".
    # i is redundant if for every j in ball_i, there exists k in code, k!=i, such that j in ball_k.
    
    # Let's implement a fast redundancy check.
    # For a given i, we need to verify that ball_i is covered by the other codewords.
    
    def is_redundant(i):
        # Check if every j in ball[i] is covered by some k in code, k != i
        for j in balls[i]:
            covered_by_other = False
            for k in code:
                if k == i:
                    continue
                if j in balls[k]: # This is O(15) check, but balls[k] is a list.
                    covered_by_other = True
                    break
            if not covered_by_other:
                return False
        return True

    # This is O(|code| * |ball| * |ball|) which is roughly 200 * 15 * 15 = 45000 ops per check.
    # 200 checks = 9M ops. Fast enough in Python? Maybe.
    
    # Let's optimize:
    # Precompute for each j, the list of codewords that cover j.
    # coverers[j] = [k for k in range(N) if j in balls[k]]
    # This is N * N memory if stored naively, but N=2187, so 2187 lists of avg size 15.
    # Total entries = 32805. Very small.
    
    coverers = [[] for _ in range(N)]
    for k in range(N):
        for j in balls[k]:
            coverers[j].append(k)
            
    def is_redundant_fast(i):
        for j in balls[i]:
            # Check if any k in coverers[j] is in code and k != i
            for k in coverers[j]:
                if k == i:
                    continue
                if k in selected_set:
                    break
            else:
                return False
        return True

    # 1-Swap:
    # Remove i, Add j (j not in code).
    # Valid if:
    # 1. ball_i \ Union(ball_k for k in code\{i}) is covered by ball_j.
    #    i.e., for any j_word in ball_i not covered by others, j_word must be in ball_j.
    # 2. ball_j covers all of ball_i? No, just the part not covered by others.
    
    # Let's refine the swap logic.
    # Let U_except_i = Union(ball_k for k in code, k!=i)
    # Let Gap = ball_i \ U_except_i
    # If we remove i and add j, the new union is U_except_i U ball_j.
    # We need (U_except_i U ball_j) to cover All.
    # Since U_except_i covers All \ Gap (because U_except_i covers All except possibly Gap),
    # we just need ball_j to cover Gap.
    
    def get_gap(i):
        # Returns list of words in ball_i not covered by any other codeword
        gap = []
        for j in balls[i]:
            covered_by_other = False
            for k in coverers[j]:
                if k == i:
                    continue
                if k in selected_set:
                    covered_by_other = True
                    break
            if not covered_by_other:
                gap.append(j)
        return gap

    # Local Search Loop
    improved = True
    while improved:
        improved = False
        # Try deletions
        for i in list(code):
            if is_redundant_fast(i):
                code.remove(i)
                selected_set.remove(i)
                improved = True
                break # Restart loop after change
        
        if improved:
            continue
            
        # Try swaps
        # For each i in code:
        #   gap = get_gap(i)
        #   If gap is empty, i is redundant (handled above)
        #   Find j not in code such that ball_j covers gap.
        #   To find j fast, we can intersect the sets of coverers for each word in gap.
        #   Candidates = Intersection(coverers[g] for g in gap)
        #   Then pick j in Candidates that is not in selected_set.
        
        for i in code:
            gap = get_gap(i)
            if not gap:
                continue
                
            # Find intersection of coverers for all words in gap
            # Start with coverers[gap[0]]
            candidates = set(coverers[gap[0]])
            for g in gap[1:]:
                candidates &= set(coverers[g])
            
            # Remove already selected codewords
            candidates -= selected_set
            
            if candidates:
                # Pick one candidate
                j = next(iter(candidates))
                # Perform swap
                code.remove(i)
                selected_set.remove(i)
                code.append(j)
                selected_set.add(j)
                improved = True
                break # Restart loop after change
                
        # Time check
        if time.time() > deadline - 5:
            break

    # Final verification and conversion
    # Verify coverage
    final_covered = [False] * N
    for i in code:
        for j in balls[i]:
            final_covered[j] = True
            
    if not all(final_covered):
        # Fallback: add missing words? Or just return what we have?
        # The task requires valid coverage. If we fail, we should try to fix or return a known good.
        # Given the time, we assume local search maintains validity if started valid.
        # But bugs might occur. Let's do a quick repair.
        for j in range(N):
            if not final_covered[j]:
                # Add a codeword that covers j
                # Pick first coverer
                for k in coverers[j]:
                    if k not in selected_set:
                        code.append(k)
                        selected_set.add(k)
                        for jj in balls[k]:
                            final_covered[jj] = True
                        break
                # If no coverer (impossible), add j itself
                if j not in selected_set:
                    code.append(j)
                    selected_set.add(j)
                    for jj in balls[j]:
                        final_covered[jj] = True

    # Convert to list of lists
    result = []
    for idx in code:
        result.append(list(all_words[idx]))
        
    return result
