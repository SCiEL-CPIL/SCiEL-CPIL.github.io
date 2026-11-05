import sys
import time
from itertools import permutations

def construct(n):
    # Classic recursive construction of length 1! + 2! + ... + n!: each permutation P of the
    # previous word, in order of appearance, becomes P m P, overlapped as far as possible.
    # We add greedy reordering at each level to maximize overlaps.
    
    start_time = time.time()
    deadline = 150.0  # seconds
    
    word = [1]
    for m in range(2, n + 1):
        # Check deadline
        if time.time() - start_time > deadline - 10:
            break
        
        # Extract all (m-1)-permutations from word in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try to reorder perms to maximize overlap
        # For small m, use greedy nearest neighbor; for larger m, use original order
        if len(perms) <= 720:  # up to 6! permutations
            # Greedy nearest neighbor reordering
            remaining = list(perms)
            if not remaining:
                new = []
            else:
                # Start with first permutation
                current = remaining.pop(0)
                ordered = [current]
                
                while remaining:
                    # Find permutation with best overlap with last in ordered
                    best_idx = 0
                    best_overlap = -1
                    last_block = list(ordered[-1]) + [m] + list(ordered[-1])
                    
                    for idx, p in enumerate(remaining):
                        block = list(p) + [m] + list(p)
                        # Calculate overlap between last_block and block
                        max_ov = min(len(last_block), len(block))
                        ov = 0
                        for o in range(max_ov, -1, -1):
                            if o == 0:
                                break
                            if last_block[len(last_block)-o:] == block[:o]:
                                ov = o
                                break
                        if ov > best_overlap:
                            best_overlap = ov
                            best_idx = idx
                    
                    chosen = remaining.pop(best_idx)
                    ordered.append(chosen)
                
                # Build new word
                new = list(ordered[0]) + [m] + list(ordered[0])
                for i in range(1, len(ordered)):
                    block = list(ordered[i]) + [m] + list(ordered[i])
                    # Find overlap
                    max_ov = min(len(new), len(block))
                    ov = 0
                    for o in range(max_ov, -1, -1):
                        if o == 0:
                            break
                        if new[len(new)-o:] == block[:o]:
                            ov = o
                            break
                    new += block[ov:]
        else:
            # For large m, use original order (too expensive to reorder)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
        
        word = new
    
    return "".join(map(str, word))
