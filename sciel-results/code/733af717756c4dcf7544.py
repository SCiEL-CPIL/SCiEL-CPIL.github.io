"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Sieve primes up to 10000
    limit = 10000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    n = len(primes)
    
    # dp[i] = list of (last_gap, length) Pareto-optimal pairs for subsequences ending at primes[i]
    # A pair (g, l) means: there's a valid subsequence ending at primes[i] with last gap g and total length l
    # Pareto: no other pair has both gap <= g and length >= l
    dp = [[] for _ in range(n)]
    
    for i in range(n):
        # Single element subsequence: no gap, length 1
        dp[i].append((0, 1))
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            for (pg, plen) in dp[j]:
                if pg <= gap:
                    # Candidate: extend subsequence ending at j to i
                    new_len = plen + 1
                    # Check if (gap, new_len) is dominated by existing entries in dp[i]
                    dominated = False
                    for (eg, el) in dp[i]:
                        if eg <= gap and el >= new_len:
                            dominated = True
                            break
                    if not dominated:
                        # Remove entries dominated by new one
                        dp[i] = [(eg, el) for (eg, el) in dp[i] if not (gap <= eg and new_len >= el)]
                        dp[i].append((gap, new_len))
        
        if time.time() > deadline:
            break
    
    # Find the maximum length
    best_len = 0
    best_i = 0
    for i in range(n):
        for (g, l) in dp[i]:
            if l > best_len:
                best_len = l
                best_i = i
    
    # Reconstruct the actual subsequence
    # We need to backtrack. Let's store parent pointers.
    # Actually, let's redo with parent tracking.
    
    # Redo with parent pointers for reconstruction
    # dp[i] = list of (last_gap, length, parent_index, parent_gap_in_parent)
    # But this gets complex. Let me just find the max and reconstruct greedily.
    
    # Actually, for the score, we just need to return a valid sequence of max length.
    # Let me reconstruct by storing choices.
    
    # Let me redo more carefully with backtracking info
    # For efficiency, let me just find max length and then reconstruct
    # by a second pass or by storing parent pointers.
    
    # Simpler: store for each (i, gap) the parent j and the gap at j
    # But that's a lot of state. Let me just do a clean reconstruction.
    
    # Actually, the evaluator just checks that the returned sequence is valid and measures its length.
    # So I need to return the actual primes.
    
    # Let me store parent pointers properly.
    # parent[i][k] = (j, gap_at_j) for the k-th entry in dp[i]
    
    # This is getting complex. Let me just use a different approach:
    # Store the full sequence for each dp entry? No, too memory heavy.
    
    # Let me just find the best length and reconstruct by searching.
    
    # For reconstruction, I'll do a second DP that stores the actual sequence.
    # Or better: store parent pointers.
    
    # Let me restart with parent pointers.
    
    # dp[i] = list of (last_gap, length, parent_idx, parent_gap)
    # where parent_idx is the index of the previous prime in the subsequence,
    # and parent_gap is the last gap of the subsequence ending at parent_idx
    # Special: if length==1, parent_idx=-1, parent_gap=0
    
    dp2 = [[] for _ in range(n)]
    
    for i in range(n):
        dp2[i].append((0, 1, -1, 0))  # single element
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            for (pg, plen, pidx, pgap) in dp2[j]:
                if pg <= gap:
                    new_len = plen + 1
                    # Check dominance
                    dominated = False
                    for (eg, el, eidx, egap) in dp2[i]:
                        if eg <= gap and el >= new_len:
                            dominated = True
                            break
                    if not dominated:
                        dp2[i] = [(eg, el, eidx, egap) for (eg, el, eidx, egap) in dp2[i] if not (gap <= eg and new_len >= el)]
                        dp2[i].append((gap, new_len, j, pg))
        
        if time.time() > deadline:
            break
    
    # Find best
    best_len = 0
    best_entry = None
    for i in range(n):
        for entry in dp2[i]:
            if entry[1] > best_len:
                best_len = entry[1]
                best_entry = (i, entry)
    
    # Reconstruct
    if best_entry is None:
        return [primes[0]] if n > 0 else []
    
    i, entry = best_entry
    seq = []
    cur_i = i
    cur_gap = entry[0]
    while cur_i != -1:
        seq.append(primes[cur_i])
        # Find the parent
        # We need to find which entry in dp2[cur_i] has last_gap == cur_gap and leads to this
        # This is tricky because there might be multiple entries with same gap
        # Let me store the full path differently.
        break
    
    # The reconstruction is messy. Let me just return the best length found as a sequence.
    # Actually, I realize I should just store the sequence directly for the best candidate.
    
    # Let me take a simpler approach: just find max length and return any valid sequence of that length.
    # For the reconstruction, let me store parent pointers more carefully.
    
    # Restart with a cleaner approach
    # Let me just use the first dp and find max length, then reconstruct by a separate search.
    
    # Actually, let me just return a valid sequence. The key is to get the right length.
    # Let me reconstruct properly.
    
    # I'll store for each dp2 entry the full chain of indices.
    # This uses more memory but n is only 1229.
    
    # Let me just do it simply: store the sequence as a tuple in each dp entry.
    
    dp3 = [[] for _ in range(n)]
    
    for i in range(n):
        dp3[i].append((0, 1, (i,)))  # (last_gap, length, sequence_of_indices)
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            for (pg, plen, pseq) in dp3[j]:
                if pg <= gap:
                    new_len = plen + 1
                    new_seq = pseq + (i,)
                    # Check dominance
                    dominated = False
                    for (eg, el, eseq) in dp3[i]:
                        if eg <= gap and el >= new_len:
                            dominated = True
                            break
                    if not dominated:
                        dp3[i] = [(eg, el, eseq) for (eg, el, eseq) in dp3[i] if not (gap <= eg and new_len >= el)]
                        dp3[i].append((gap, new_len, new_seq))
        
        if time.time() > deadline:
            break
    
    # Find best
    best_len = 0
    best_seq = ()
    for i in range(n):
        for (g, l, s) in dp3[i]:
            if l > best_len:
                best_len = l
                best_seq = s
    
    if not best_seq:
        return [primes[0]] if n > 0 else []
    
    return [primes[idx] for idx in best_seq]

# EVOLVE-BLOCK-END
