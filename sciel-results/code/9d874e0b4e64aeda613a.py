import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Use a greedy algorithm with multiple random restarts and different orderings
    # or a backtracking search.
    # Given the small size, backtracking is feasible.
    
    best_blocks = []
    
    # We can use the known construction if we can generate S(4,5,11) reliably.
    # Let's try to generate S(4,5,11) using a small backtracker first.
    # If that succeeds, we build S(5,6,12).
    
    def find_s4511():
        # Backtracking for S(4,5,11)
        # Points 0..10
        # Target 66 blocks
        covered_4 = set()
        blocks = []
        
        # Precompute all 5-subsets
        all_5 = list(combinations(range(11), 5))
        
        # To speed up, we can use a heuristic ordering
        # But for 11 points, it's small enough.
        
        def backtrack(idx, blocks, covered_4):
            if time.time() > deadline:
                return None
            if len(blocks) == 66:
                return blocks
            
            # Find the first uncovered 4-subset? No, we need to cover all.
            # Better: iterate through candidate 5-subsets.
            
            # Heuristic: Pick the 5-subset that has the most conflicts?
            # Or just lexicographic.
            
            for b in all_5[idx:]:
                # Check if b conflicts with covered_4
                # b has 5 4-subsets
                b_4s = list(combinations(b, 4))
                if any(s in covered_4 for s in b_4s):
                    continue
                
                # Try adding b
                new_covered = covered_4.union(b_4s)
                res = backtrack(idx+1, blocks + [b], new_covered)
                if res:
                    return res
            
            return None
        
        res = backtrack(0, [], set())
        return res
    
    s4511 = find_s4511()
    
    if s4511:
        # Construct S(5,6,12)
        blocks_12 = []
        for b in s4511:
            # Block containing 11
            blocks_12.append(tuple(list(b) + [11]))
            # Block not containing 11 (complement in 0..10)
            comp = tuple(sorted(set(range(11)) - set(b)))
            blocks_12.append(comp)
        return [list(b) for b in blocks_12]
    else:
        # Fallback to greedy if backtracking fails/times out
        # (Unlikely to be optimal, but better than nothing)
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            if time.time() > deadline:
                break
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
