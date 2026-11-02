import itertools
import time
import sys

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 4, returns the known optimal.
    For n >= 5, uses block-based TSP with 2-opt local search.
    """
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341214312431423413241342143214"
    
    # For n >= 5, use block-based approach
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    num_blocks = len(prev_perms)
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
    
    # Convert to strings for fast overlap computation
    block_strs = ["".join(map(str, b)) for b in blocks]
    block_len = len(blocks[0])  # 2*(n-1)+1
    
    # Precompute overlaps: overlaps[i][j] = max k such that block_strs[i][-k:] == block_strs[j][:k]
    # For efficiency, we can limit k to block_len (max possible)
    overlaps = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = block_strs[i]
        for j in range(num_blocks):
            if i == j:
                overlaps[i][j] = block_len  # Self-overlap is full
                continue
            bj = block_strs[j]
            # Max meaningful overlap: we can't overlap more than block_len
            # But structurally, for p_i + [n] + p_i and p_j + [n] + p_j,
            # overlap > n-1 would require the [n] to align, which is impossible for distinct i,j
            # So max overlap is n-1 (if p_i suffix matches p_j prefix of length n-1)
            # Actually overlap can be up to 2*(n-1) in worst case but let's just check up to block_len
            max_k = min(block_len, 2 * (n - 1))
            found = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found
    
    # Get the classic construction order as initial path
    # The classic construction orders blocks by the order their (n-1)-permutations appear
    # in the superpermutation of n-1 symbols.
    # For simplicity, use the natural order from itertools.permutations as initial path,
    # then improve with 2-opt.
    
    # Actually, let's use the classic recursive construction to get a valid ordering
    # that's known to be good, then refine with 2-opt.
    
    def classic_order(n):
        """Get the order of (n-1)-perms as they appear in the classic n-superperm."""
        if n <= 2:
            return [tuple(range(1, n))]
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        # Extract the order of (n-1)-perms from the final word
        # In the classic construction, each block is p + [n] + p
        # The (n-1)-perm p appears as the prefix and suffix of each block
        # We can extract them by scanning for [n]
        order = []
        seen = set()
        for i in range(len(word) - n + 1):
            if word[i + n - 1] == n:  # [n] is at position i+n-1? No...
                pass
        # Simpler: just use the perms list from the last iteration
        # Re-derive: in the last step (m=n), perms contains the (n-1)-perms in order
        word2 = [1]
        perms_list = []
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word2) - m + 2):
                window = tuple(word2[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            if m == n:
                perms_list = perms
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word2 = new
        return perms_list
    
    # Get classic order
    classic_perms = classic_order(n)
    
    # Map permutations to block indices
    perm_to_idx = {p: i for i, p in enumerate(prev_perms)}
    initial_path = [perm_to_idx[p] for p in classic_perms]
    
    def calc_length(path):
        total = block_len
        for k in range(len(path) - 1):
            total += block_len - overlaps[path[k]][path[k + 1]]
        return total
    
    # 2-opt local search
    best_path = initial_path[:]
    best_len = calc_length(best_path)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(len(best_path) - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 2, len(best_path)):
                # Compute delta in total length for reversing path[i+1:j+1]
                # Old overlaps involved:
                #   overlaps[path[i]][path[i+1]]
                #   overlaps[path[j]][path[j+1]] if j+1 < len
                #   internal: overlaps[path[k]][path[k+1]] for k in [i+1, j-1]
                # New overlaps:
                #   overlaps[path[i]][path[j]]
                #   overlaps[path[i+1]][path[j+1]] if j+1 < len
                #   internal: overlaps[path[k+1]][path[k]] for k in [i+1, j-1]
                
                old_ov = overlaps[best_path[i]][best_path[i + 1]]
                new_ov = overlaps[best_path[i]][best_path[j]]
                
                if j + 1 < len(best_path):
                    old_ov += overlaps[best_path[j]][best_path[j + 1]]
                    new_ov += overlaps[best_path[i + 1]][best_path[j + 1]]
                
                for k in range(i + 1, j):
                    old_ov += overlaps[best_path[k]][best_path[k + 1]]
                    new_ov += overlaps[best_path[k + 1]][best_path[k]]
                
                if new_ov > old_ov:
                    best_path[i + 1:j + 1] = reversed(best_path[i + 1:j + 1])
                    improved = True
                    current_len = best_len - (new_ov - old_ov)
                    if current_len < best_len:
                        best_len = current_len
    
    # Build final word
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        ov = overlaps[best_path[k - 1]][best_path[k]]
        final_word += blocks[best_path[k]][ov:]
    
    return "".join(map(str, final_word))
