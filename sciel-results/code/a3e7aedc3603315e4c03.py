import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate permutations of 1..n-1
    # For n=7, this is 1..6
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    
    # Standard sequence: order of appearance in the standard superpermutation of n-1
    # We can generate the standard superpermutation of n-1 and extract the distinct windows
    def standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            p_list, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    p_list.append(window)
            new = []
            for p in p_list:
                block = list(p) + [k] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    # Get standard sequence for n-1
    # For n=7, m=6
    if n == 2:
        return "121"
    
    w_prev = standard_superperm(n - 1)
    
    # Extract distinct windows of size n-1 from w_prev
    seq = []
    seen = set()
    for i in range(len(w_prev) - (n - 1) + 1):
        window = tuple(w_prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            seq.append(window)
            
    # seq is a list of tuples, each a permutation of 1..n-1
    # We want to reorder seq to maximize sum of overlaps
    # Overlap between p and q is length of longest suffix of p that is prefix of q
    
    def overlap(p, q):
        max_ov = 0
        # p and q are tuples of length n-1
        # Check suffix of p of length k against prefix of q of length k
        # Max possible is n-2? No, n-1.
        # If p == q, overlap is n-1.
        for k in range(len(p) - 1, -1, -1):
            if p[-k:] == q[:k]:
                return k
        return 0

    # Initial total overlap
    current_seq = seq
    total_ov = sum(overlap(current_seq[i], current_seq[i+1]) for i in range(len(current_seq) - 1))
    
    # Local search
    # Time limit check
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_seq) - 1):
            if time.time() > deadline:
                break
            # Try swap i, i+1
            p_i = current_seq[i]
            p_ip1 = current_seq[i+1]
            
            # Calculate change
            old_part = 0
            new_part = 0
            
            if i > 0:
                old_part += overlap(current_seq[i-1], p_i)
                new_part += overlap(current_seq[i-1], p_ip1)
            
            old_part += overlap(p_i, p_ip1)
            new_part += overlap(p_ip1, p_i)
            
            if i < len(current_seq) - 2:
                old_part += overlap(p_ip1, current_seq[i+2])
                new_part += overlap(p_i, current_seq[i+2])
                
            if new_part > old_part:
                # Swap
                current_seq[i], current_seq[i+1] = p_ip1, p_i
                total_ov += new_part - old_part
                improved = True
                # Restart scan? Or continue? Continue is faster.
                # But might miss better swaps.
                # Given small improvement, simple pass might suffice.
                # Let's do a full pass until no improvement.

    # Construct final word
    word = []
    for i, p in enumerate(current_seq):
        block = list(p) + [n] + list(p)
        if i == 0:
            word = block
        else:
            prev_p = current_seq[i-1]
            ov = overlap(prev_p, p)
            # Append block[ov:]
            word += block[ov:]
            
    return "".join(map(str, word))
