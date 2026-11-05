import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin
    
    if n <= 1:
        return "1"
    
    # Step 1: Construct the superpermutation for n-1
    # We use the standard recursive construction for the base case to ensure we have all (n-1)! permutations
    # Note: For n=7, we need the word for n=6.
    
    def build_base(k):
        if k == 1:
            return [1]
        word = build_base(k - 1)
        m = k
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap with current end of new
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0:
                    max_overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        return new

    # Get the word for n-1
    base_word = build_base(n - 1)
    
    # Step 2: Extract the (n-1)! permutations from base_word
    m = n - 1
    perms, seen = [], set()
    for i in range(len(base_word) - m + 1):
        window = tuple(base_word[i:i + m])
        if window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Step 3: Create blocks B_i = P_i + [n] + P_i
    # There are n! blocks
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
        
    num_blocks = len(blocks)
    
    # Step 4: Compute overlap matrix
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # We only need to check up to m (n-1) overlaps, because the middle [n] prevents larger overlaps
    # Actually, the max possible overlap is m (n-1).
    # Let's verify: B_i = P_i n P_i. B_j = P_j n P_j.
    # Suffix of B_i ends with P_i. Prefix of B_j starts with P_j.
    # If P_i == P_j, overlap is m.
    # If P_i suffix matches P_j prefix, overlap can be less.
    # Max overlap is indeed m.
    
    # Optimization: Precompute suffixes and prefixes?
    # For 5040 blocks, 5040*5040*6 checks is ~150M operations. Might be slow in Python.
    # Let's try to speed up overlap calculation.
    
    # Alternative: Map permutations to indices.
    # Overlap between B_i and B_j depends on P_i and P_j.
    # Let's just compute it directly but efficiently.
    
    # To save time, we can use a greedy approach without full matrix if it's too slow,
    # but 5040 is small enough for a sparse or cached approach?
    # Let's try computing the matrix in a tight loop.
    
    # Actually, we can observe that the overlap between B_i and B_j is determined by
    # the longest suffix of P_i that is a prefix of P_j.
    # Let LCP(P_i_suffix, P_j_prefix).
    
    # Let's precompute for each permutation P, its "prefixes" and "suffixes" maps?
    # Or just compute on the fly with a limit.
    
    # Let's try a heuristic search directly on the sequence of blocks.
    # We start with a random or greedy order and improve.
    
    # Greedy Initialization:
    # Start with any block.
    # Repeatedly pick the next block that maximizes overlap with the current end.
    # This is O(N^2) per step, O(N^3) total. 5040^3 is too slow.
    
    # Better Greedy:
    # Maintain a set of unused blocks.
    # Current end is a string (or tuple).
    # For each unused block, calculate overlap.
    # This is still O(N^2 * M) where M is overlap calc cost.
    
    # Let's try to build the overlap matrix using NumPy or just Python lists with optimization.
    # 5040 blocks.
    # We can represent permutations as tuples of ints.
    
    # Let's calculate the overlap matrix M.
    # M[i][j] is int.
    
    # To speed up, note that if P_i and P_j share no common prefix/suffix structure, overlap is 0 or 1.
    # Most overlaps will be small.
    
    # Let's attempt a 2-Opt search on a greedy solution.
    # 1. Greedy construction of the order.
    # 2. 2-Opt improvement.
    
    # Greedy Construction:
    # Start with block 0.
    # While unused blocks remain:
    #   Find j in unused that maximizes overlap(current_end, blocks[j]).
    #   Append blocks[j].
    #   Update current_end.
    
    # To make this fast, we need fast overlap calc.
    # Overlap(A, B) where A is suffix of current word, B is prefix of next block.
    # Actually, we just need overlap between the *last block* and the *candidate block*.
    # Let last_block be the block we just added.
    # We need max k such that last_block[-k:] == candidate_block[:k].
    
    # Let's precompute for each block its prefixes and suffixes?
    # There are 5040 blocks.
    # For each block, store a dictionary: suffix_tuple -> length? No, suffix is unique for a given length.
    # We can store: for each block b, a dict `pref_map` where key is prefix of length L, value is L?
    # No, we need to match suffix of A with prefix of B.
    
    # Let's just compute overlaps on the fly in the greedy loop, but use a heuristic to prune.
    # Or, since 150s is generous, maybe we can afford a slower greedy?
    # 5040 steps. In each step, check 5040 candidates.
    # 25 million overlap checks.
    # Each check is up to 6 comparisons.
    # 150 million comparisons. Python might take ~10-20 seconds. This is feasible.
    
    # Let's implement the Greedy method.
    
    used = [False] * num_blocks
    order = [0]
    used[0] = True
    current_block = blocks[0]
    
    # We need to track the actual suffix of the growing word to handle overlaps correctly?
    # No, the overlap is only between the *last added block* and the *next block*.
    # Because the internal structure of the superpermutation ensures that no "cross-block" overlap
    # (spanning more than 2 blocks) is possible or beneficial in this construction scheme?
    # Wait, the standard construction relies on the fact that the overlaps are local.
    # Yes, in the recursive superpermutation, the word is formed by concatenating blocks with local overlaps.
    # So we just need to maximize the sum of pairwise overlaps between consecutive blocks in the sequence.
    
    # Greedy Loop
    for _ in range(num_blocks - 1):
        best_idx = -1
        best_overlap = -1
        # Check all unused blocks
        for j in range(num_blocks):
            if not used[j]:
                # Calculate overlap between current_block and blocks[j]
                b_j = blocks[j]
                max_ov = 0
                # Max possible overlap is m (n-1)
                for k in range(m, 0, -1):
                    if current_block[-k:] == b_j[:k]:
                        max_ov = k
                        break
                if max_ov > best_overlap:
                    best_overlap = max_ov
                    best_idx = j
        
        if best_idx == -1:
            # Should not happen
            break
            
        used[best_idx] = True
        order.append(best_idx)
        current_block = blocks[best_idx]
        
        # Time check
        if time.time() > deadline - 10:
            break

    # If we didn't finish greedy, we have a partial order.
    # We must complete it arbitrarily to have a valid permutation of blocks.
    if len(order) < num_blocks:
        remaining = [i for i in range(num_blocks) if not used[i]]
        order.extend(remaining)

    # Now we have an order of blocks.
    # Let's try 2-Opt improvement to see if we can reduce length further.
    # 2-Opt: Reverse a segment of the order.
    # Calculate total length of current order.
    
    def calc_length(order_list):
        if not order_list:
            return 0
        total = len(blocks[order_list[0]])
        for i in range(len(order_list) - 1):
            b1 = blocks[order_list[i]]
            b2 = blocks[order_list[i+1]]
            ov = 0
            for k in range(m, 0, -1):
                if b1[-k:] == b2[:k]:
                    ov = k
                    break
            total += len(b2) - ov
        return total

    current_len = calc_length(order)
    
    # 2-Opt Loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Randomize start point or just iterate
        # To be fast, we can try a few random reversals or iterate through all pairs (too slow for 5040).
        # 5040^2 / 2 = 12.5M pairs. Too many for Python in 150s if we recalc length from scratch.
        # We need incremental delta calculation.
        
        # Delta for reversing [i, j]:
        # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
        # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, i-1)
        # We need overlap(i-1, i), overlap(i-1, j), overlap(j, j+1), overlap(i, j+1)
        # And the internal overlaps of the reversed segment are just the reverse of the forward ones?
        # No, overlap(A, B) != overlap(B, A) generally.
        # So we need to sum overlaps for the new internal edges.
        # This is O(length of segment) per move.
        
        # Given time constraints, let's stick to the Greedy result if 2-Opt is too complex to implement efficiently in the remaining time.
        # The Greedy result is usually quite good.
        
        # Let's just do a few random 2-Opt moves if time permits, with fast delta check.
        # Actually, let's just return the greedy result. It's a solid improvement over the fixed recursive order if the fixed order is bad.
        # The standard recursive order is deterministic. Greedy might find a better one.
        
        pass # Skip complex 2-Opt for now to ensure stability and speed.

    # Assemble the word
    final_word = []
    # Add first block
    final_word.extend(blocks[order[0]])
    
    for i in range(1, len(order)):
        prev_block = blocks[order[i-1]]
        curr_block = blocks[order[i]]
        
        ov = 0
        for k in range(m, 0, -1):
            if prev_block[-k:] == curr_block[:k]:
                ov = k
                break
        final_word.extend(curr_block[ov:])
        
    return "".join(map(str, final_word))

# Helper for testing/verification (not used in construct)
def verify(word, n):
    s = set()
    for i in range(len(word) - n + 1):
        window = tuple(int(c) for c in word[i:i+n])
        s.add(window)
    expected = set(itertools.permutations(range(1, n+1)))
    return s == expected

if __name__ == "__main__":
    # Quick test for small n
    for n in range(1, 6):
        w = construct(n)
        # print(f"n={n}, len={len(w)}, valid={verify(w, n)}")
        pass
