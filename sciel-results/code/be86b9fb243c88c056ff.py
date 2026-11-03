import time
from collections import Counter
import heapq

def construct(n):
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
        
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        # The seed code does this by scanning windows of length m-1
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Greedy Deletion with priority queue
    seq = word
    L = len(seq)
    
    # Initialize counts
    counts = Counter()
    for i in range(L - n + 1):
        w = tuple(seq[i:i+n])
        counts[w] += 1
    
    # Priority queue: (priority, index) - lower priority means more deletable
    # Priority: number of windows containing this index that have count > 1
    # We want to delete positions where all affected windows have count > 1
    
    def can_delete(i):
        """Check if deleting position i is valid and return affected windows."""
        start_min = max(0, i - n + 1)
        start_max = min(len(seq) - n, i)
        
        if start_min > start_max:
            return True, []
        
        affected = []
        for start in range(start_min, start_max + 1):
            w = tuple(seq[start:start+n])
            if counts[w] == 1:
                return False, []
            affected.append(w)
        return True, affected
    
    # Build initial priority queue
    pq = []
    for i in range(len(seq)):
        valid, affected = can_delete(i)
        if valid:
            # Priority: number of affected windows (fewer is better to delete first)
            heapq.heappush(pq, (len(affected), i))
    
    while pq:
        if time.time() - start_time > deadline:
            break
        
        _, i = heapq.heappop(pq)
        
        # Check if this deletion is still valid
        valid, affected = can_delete(i)
        if not valid:
            continue
        
        # Perform deletion
        for w in affected:
            counts[w] -= 1
            if counts[w] == 0:
                del counts[w]
        
        seq.pop(i)
        
        # Update counts for new windows formed by merge
        new_start_min = max(0, i - n + 1)
        new_start_max = min(len(seq) - n, i)
        
        for start in range(new_start_min, new_start_max + 1):
            w = tuple(seq[start:start+n])
            counts[w] += 1
        
        # Add new candidate positions to the queue
        # Positions near the deletion point might now be deletable
        for j in range(max(0, i - n), min(len(seq), i + n)):
            if j < i:
                # Indices shifted left by 1 after deletion
                pass
            valid, affected = can_delete(j)
            if valid:
                heapq.heappush(pq, (len(affected), j))
    
    return "".join(map(str, seq))
