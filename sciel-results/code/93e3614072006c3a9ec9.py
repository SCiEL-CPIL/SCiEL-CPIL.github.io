import time
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    best_sequence = []
    best_len = 0
    
    # We will use a recursive backtracking search.
    # State: current list of values, current set of all consecutive block sums.
    # Optimization: Precompute potential sums? No, sums depend on the specific sequence.
    # We need to efficiently check if adding a new value 'x' creates duplicate sums.
    
    # To speed up, we can represent the sequence and sums efficiently.
    # Let's use a set for sums.
    # When we append x to seq [a1, ..., ak], the new sums are:
    # x, x + ak, x + ak + a{k-1}, ..., x + sum(seq)
    # These are x + suffix_sums.
    
    # We want to maximize length.
    # Heuristic: Try larger numbers first? Or numbers that keep the total sum small?
    # Actually, Erdos-Harzheim suggests powers of 2 work well (length ~ log(N)).
    # Greedy often yields length around 12-14 for N=256?
    # Let's check what the greedy seed does.
    # Seed: 
    # x=1, sums={1}
    # x=2, tails={2, 3}, sums={1,2,3}
    # x=3, tails={3,5,6} -> 3 exists. Skip.
    # x=4, tails={4,8,12} -> OK. seq=[1,2,4], sums={1,2,3,4,5,6,8,12}
    # ...
    
    # A smarter search might find length 15 or 16.
    
    def backtrack(seq, sums, current_len):
        nonlocal best_sequence, best_len, start_time
        
        if time.time() - start_time > time_limit:
            return False
        
        if current_len > best_len:
            best_len = current_len
            best_sequence = list(seq)
        
        # Pruning: If we can't possibly beat best_len, stop.
        # Max possible length is bounded by number of distinct sums possible.
        # Total sum S. Max number of blocks is n(n+1)/2.
        # Distinct sums must be in [1, S_max].
        # This bound is loose.
        
        # Try candidates from 1 to 256.
        # Heuristic: To maximize length, we generally want small numbers to keep sums small,
        # BUT small numbers cause collisions quickly.
        # Powers of 2 are collision-free.
        # Let's try candidates in increasing order, but maybe skip some?
        # Or try increasing order, as it's the most likely to allow *many* steps?
        # Actually, if we pick large numbers early, we run out of range [1,256] quickly.
        # So we MUST pick small numbers early to fit many numbers in [1,256].
        # Therefore, trying smaller candidates first is a good heuristic for maximizing *count*.
        
        # However, the greedy approach already does this. Why did it fail to be optimal?
        # Because it commits. Backtracking allows us to pick a slightly larger number early
        # if it prevents a dead end later, allowing a longer sequence overall.
        
        # Let's iterate candidates from 1 to 256.
        # To improve on greedy, we can stop trying candidates if the remaining
        # potential length + current_len <= best_len.
        
        # How many more elements can we add?
        # We have used 'current_len' elements.
        # The minimum possible sum of the next k elements (assuming they are 1,2,...) 
        # is not a tight bound because they must be > last element and distinct.
        
        # Simple bound: We have 256 - max(seq) numbers left available? No, we can reuse gaps.
        # But they must be strictly increasing.
        # So the next element must be > seq[-1].
        # The smallest possible next element is seq[-1] + 1.
        # The smallest possible sequence of length L starting after seq[-1] is:
        # seq[-1]+1, seq[-1]+2, ..., seq[-1]+L
        # So max possible additional length is 256 - seq[-1].
        
        if 256 - seq[-1] <= best_len - current_len:
            return False
        
        last_val = seq[-1] if seq else 0
        total_sum = sum(seq)
        
        # Generate suffix sums of current seq to check new sums quickly
        # suffix_sums[i] = sum(seq[i:])
        # New sums when adding x: x + suffix_sums[i] for i in 0..len(seq)
        # Note: suffix_sums[0] = total_sum, suffix_sums[len(seq)] = 0 (conceptually for just x)
        # Actually, the new blocks are x, x+last, x+last+prev...
        # So we need to check if x + s is in sums for any suffix sum s of seq (including 0? No, 0 is not a sum of a non-empty block of seq, but x+0=x is a block).
        # Let's maintain a list of suffix sums including 0 at the end for convenience.
        
        # To avoid recomputing suffix sums every step, we can pass them down.
        # But for simplicity in this first iteration, let's compute them.
        # Optimization: Pass suffix_sums list.
        
        # Let's refactor to pass suffix_sums.
        # This function signature change is internal.
        
        # For now, simple loop.
        # We can optimize the candidate loop.
        
        # Let's try candidates from last_val + 1 to 256.
        # If we find a valid one, recurse.
        
        # To make it different from greedy: Greedy stops at the first valid one.
        # Backtracking tries ALL valid ones if the first one doesn't lead to a better solution.
        
        # Heuristic for ordering:
        # If we pick a number that is "too close" to previous ones, we might run out of sum space.
        # If we pick a number that is "too far", we run out of value range.
        # Let's stick to increasing order for candidates, as it maximizes the count of remaining values.
        
        for x in range(last_val + 1, 257):
            if time.time() - start_time > time_limit:
                return False
            
            # Check validity
            # New sums: x, x + seq[-1], x + seq[-1] + seq[-2], ...
            # We need to check if any of these are in 'sums'.
            
            # Compute new sums
            # s = x
            # if s in sums: continue
            # s += seq[-1]
            # if s in sums: continue
            # ...
            
            valid = True
            s = x
            if s in sums:
                valid = False
            else:
                # Add x to the end of seq
                # The new sums are x + suffix_sums of seq
                # Let's iterate through seq in reverse to build the sums
                # This is O(n) per candidate.
                # n is small (< 20). Candidates < 256. 20*256 = 5120 ops per node.
                # Depth 20. 5120 * 20^20 is too big.
                # We need better pruning.
                
                # Let's compute the sums and check.
                # We can break early if a collision is found.
                
                # Re-do check efficiently
                # We already checked x.
                # Now check x + seq[-1], x + seq[-1] + seq[-2], etc.
                
                # Let's use the suffix sum list passed down? 
                # I didn't pass it in the signature above. 
                # Let's modify the inner function to take suffix_sums.
                
                # I will rewrite the backtrack function to be more efficient.
                
                pass # Placeholder, will implement properly below

        return False

    # Proper implementation with suffix sums
    def bt(seq, sums, suffix_sums):
        nonlocal best_sequence, best_len, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_sequence = list(seq)
        
        if not seq:
            last_val = 0
        else:
            last_val = seq[-1]
            
        # Pruning: Max possible additional elements
        if 256 - last_val <= best_len - current_len:
            return
            
        # Try candidates
        for x in range(last_val + 1, 257):
            if time.time() - start_time > time_limit:
                return
            
            # Check collisions
            # New sums are x + s for s in [0] + suffix_sums
            # suffix_sums contains sums of suffixes of seq.
            # e.g. seq=[a,b], suffix_sums=[b, a+b, 0] ? No.
            # Let's define suffix_sums[i] = sum(seq[i:]).
            # Then new sums are x + suffix_sums[i] for i in 0..len(seq)-1, and x itself (which is x+0).
            
            valid = True
            # Check x
            if x in sums:
                valid = False
            else:
                # Check x + suffix sums
                # We need to add x to each suffix sum
                # And check if the result is in sums
                for s in suffix_sums:
                    new_sum = x + s
                    if new_sum in sums:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Valid candidate
            # Prepare new state
            new_seq = seq + [x]
            new_sums = sums.copy()
            new_sums.add(x)
            
            # Update suffix sums
            # Old suffix sums: [sum(seq[0:]), sum(seq[1:]), ..., sum(seq[-1:])]
            # New suffix sums for new_seq:
            # [x + sum(seq[0:]), x + sum(seq[1:]), ..., x + sum(seq[-1:]), x]
            # Note: The order in the list doesn't matter for the set of sums,
            # but for the next recursion, we need the correct suffix sums of the NEW seq.
            # new_suffix_sums = [x + s for s in suffix_sums] + [x]
            
            new_suffix_sums = [x + s for s in suffix_sums]
            new_suffix_sums.append(x)
            
            # Add new sums to set
            for s in new_suffix_sums:
                new_sums.add(s)
                
            # Recurse
            bt(new_seq, new_sums, new_suffix_sums)
            
            # Backtrack (implicit in functional style, but we modify local vars in loop)
            # No need to explicitly remove from 'sums' as we passed a copy.
            # However, copying a set of size O(n^2) is expensive.
            # n is small, but depth is ~15.
            # Set size ~ 120. Copying 120 elements 15 levels deep is fine.
            
            # Optimization: If we found a solution of length best_len, 
            # and we are exploring candidates in increasing order, 
            # do we need to explore further?
            # Yes, because a larger x might allow a longer sequence later?
            # Actually, larger x reduces the range for future elements.
            # So smaller x is generally better for length.
            # If we found a valid x, and we recurse, and it fails to beat best_len,
            # should we try the next x?
            # Yes.
            
            # Early termination?
            # If current_len + 1 >= best_len, we are trying to beat it.
            # If we find a valid x, we must explore it.
            
    # Initial call
    # Empty sequence
    # sums = set()
    # suffix_sums = [] (empty list, represents no suffixes)
    # Wait, for empty seq, suffix_sums should be []?
    # If seq=[], new sums for x is just {x}.
    # My loop: for s in suffix_sums (empty), so only checks x. Correct.
    # new_suffix_sums for empty seq + [x]:
    # [x + s for s in []] + [x] = [x]. Correct.
    
    bt([], set(), [])
    
    return best_sequence

# Quick test to ensure it doesn't crash and returns something
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    
    # Verify
    sums = set()
    valid = True
    for i in range(len(res)):
        s = 0
        for j in range(i, len(res)):
            s += res[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    if valid:
        print("Valid")
    else:
        print("Invalid")
