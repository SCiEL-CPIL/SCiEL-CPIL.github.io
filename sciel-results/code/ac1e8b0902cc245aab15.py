import time
from collections import Counter
import random

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
        
    # 1. Generate standard superpermutation
    # Recursive construction
    def generate_superpermutation(n):
        if n == 1:
            return [1]
        prev = generate_superpermutation(n - 1)
        res = []
        for i in range(1, n + 1):
            block = prev + [i]
            # Overlap with last part of res
            # The standard construction appends the new symbol and then the previous superpermutation
            # Wait, the standard recursive construction is:
            # S_n = S_{n-1} 1 S_{n-1} 2 ... S_{n-1} n
            # With overlaps.
            # Actually, the length formula $L_n = n L_{n-1} - (n-1)$ implies:
            # S_n is formed by concatenating $S_{n-1}$ with $i$ prepended/appended?
            # Let's check n=2: S_1=[1]. S_2 = [1,2,1] (len 3). Formula: 2*1 - 1 = 1? No.
            # L_1=1. L_2=3. L_3=9.
            # S_2 = 1 2 1.
            # S_3 = 1 2 1 3 1 2 1 3 2 1 3? No.
            # S_3 = 1 2 1 3 1 2 1 3 2 1 3 2 1?
            # Let's use the known algorithm:
            # S_n = S_{n-1}(1) S_{n-1}(2) ... S_{n-1}(n)
            # Where S_{n-1}(k) is S_{n-1} with k inserted?
            
            # Actually, the standard construction is:
            # S_n = S_{n-1} 1 S_{n-1} 2 ... 
            # No, it's:
            # S_n = S_{n-1} (1) S_{n-1} (2) ...
            # Let's just use the iterative method from the seed, it was correct.
            pass
        return res

    # Re-verify seed's generation logic
    # The seed logic:
    # word = [1]
    # for m in 2..n:
    #   extract unique perms of len m-1
    #   for each perm p:
    #     block = p + [m] + p
    #     append with max overlap
    
    # This is correct and generates the standard superpermutation.
    
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    seq = word
    L = len(seq)
    
    # Greedy Deletion
    # We will try to delete indices.
    # To be efficient, we maintain a list of "active" indices to check.
    # Initially all indices are candidates.
    
    # Helper to check if index i is deletable
    # We need window counts.
    
    def calculate_counts(seq, n):
        counts = Counter()
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i:i+n])] += 1
        return counts

    counts = calculate_counts(seq, n)
    
    # List of indices to try. We will try to delete from left to right, 
    # but if we delete an index, the indices to the right shift.
    # So we restart the scan or adjust.
    # Given L ~ 6000, a full scan is O(L). 
    # If we delete, we do O(L) work to update counts (actually only local).
    # But checking all L indices is O(L * n) if we check each window.
    # We can precompute "deletability" for each index.
    
    # deletable[i] = True if all windows containing i have count > 1.
    
    def update_deletability(seq, counts, i, n):
        # Returns list of indices in range [i-n+1, i] that are affected
        # And updates their status? 
        # This is complex. 
        # Simpler: Just check on the fly.
        pass

    # Strategy: 
    # 1. Compute counts.
    # 2. Create a list `candidates` of all indices.
    # 3. Iterate through candidates.
    # 4. For each candidate i, check if deletable.
    # 5. If yes, delete it, update counts, and restart the candidate list? 
    #    Or just continue? 
    #    If we delete i, indices > i shift left. 
    #    It's safer to restart the scan from 0 after each deletion.
    #    Worst case: L deletions * L checks = L^2 = 36M operations.
    #    In Python, 36M simple ops might take ~1-2 seconds. 
    #    This is well within 110s.
    
    # Let's implement the "Restart on Deletion" strategy.
    
    while True:
        if time.time() - start_time > deadline:
            break
            
        deleted_any = False
        # We scan from left to right.
        # We need to check indices 0 to len(seq)-1.
        # If we delete at i, we restart.
        
        for i in range(len(seq)):
            # Check if i is deletable
            # Windows containing i: start from max(0, i-n+1) to min(len(seq)-n, i)
            start_min = max(0, i - n + 1)
            start_max = min(len(seq) - n, i)
            
            deletable = True
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                if counts[w] <= 1:
                    deletable = False
                    break
            
            if deletable:
                # Perform deletion
                # Update counts for windows that are removed
                for start in range(start_min, start_max + 1):
                    w = tuple(seq[start:start+n])
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                # Remove element
                seq.pop(i)
                
                # Update counts for new windows formed
                # New windows start from max(0, i-n+1) to min(len(seq)-n, i)
                # Note: len(seq) decreased by 1.
                new_start_min = max(0, i - n + 1)
                new_start_max = min(len(seq) - n, i)
                
                for start in range(new_start_min, new_start_max + 1):
                    w = tuple(seq[start:start+n])
                    counts[w] += 1
                    
                deleted_any = True
                break # Restart scan
        
        if not deleted_any:
            break
            
    return "".join(map(str, seq))
