"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a strictly increasing sequence of integers in [1, 256] such that
    sums of all nonempty consecutive blocks are distinct. Maximize length.
    
    Strategy:
    1. Use a greedy approach to get a good initial solution.
    2. Use backtracking with forward checking to try to improve.
    """
    deadline = time.time() + 80  # Leave 10s safety margin for 90s limit
    
    # First, get a greedy solution
    greedy_seq = _greedy_construct()
    best_seq = list(greedy_seq)
    best_len = len(best_seq)
    
    if best_len == 0:
        return best_seq
    
    # Try to improve with backtracking
    # We'll try to build sequences of length > best_len
    # Use iterative deepening: try to find sequences of length best_len + 1, best_len + 2, etc.
    
    # For efficiency, we use a bitset for sums
    # Max possible sum: 256 * 256 = 65536 (if all elements were 256, but they're distinct and increasing)
    # Actually max sum of any block: sum of all elements. If we have k elements up to 256, max sum ~ k*256
    # Let's use a set for sums for simplicity, but with forward checking
    
    # Let's try a smarter approach:
    # Start with the greedy sequence and try to modify it
    # Or start from scratch with backtracking and pruning
    
    # Backtracking approach with forward checking
    # We build the sequence element by element
    # At each step, we try candidates from last+1 to 256
    # We check if adding the candidate creates any duplicate block sums
    # Forward checking: after adding, we check if there are enough remaining candidates
    # to reach the target length
    
    # Let's implement a DFS that tries to extend beyond the greedy length
    # We'll use the greedy sequence as a starting point and try to find a longer one
    
    # Actually, let's just do a full backtracking search with good pruning
    # The key insight: the number of block sums for a sequence of length n is n*(n+1)/2
    # These must all be distinct, so we need n*(n+1)/2 distinct sums
    # The sums range from 1 (min element) to sum(all elements)
    # For n elements in [1, 256], the max sum is at most 256+255+...+(256-n+1)
    
    # Let's estimate the max possible length:
    # If we have n elements, we have n*(n+1)/2 block sums
    # The smallest possible block sum is 1, largest is at most ~n*256
    # So we need n*(n+1)/2 <= ~n*256, which gives n+1 <= 512, so n <= 511
    # But that's a loose bound. The actual bound is tighter.
    
    # From the greedy solution, we know we can get at least best_len
    # Let's try to find a longer sequence using backtracking
    
    # For backtracking, we need efficient sum collision checking
    # When we add a new element x to sequence [a1, a2, ..., ak], the new block sums are:
    # x, x+ak, x+ak+a_{k-1}, ..., x+ak+...+a1
    # These are k+1 new sums. We need to check if any of these are in the existing set of sums.
    
    # Let's use a set for the sums. The number of sums is at most n*(n+1)/2.
    # For n ~ 20, that's 210 sums. A set lookup is O(1) amortized.
    
    # The main cost is the branching factor: at each step, we try up to 256 - last candidates.
    # With good pruning, this should be manageable.
    
    # Let's implement the backtracking:
    
    def _dfs(sequence, sums_set, last_val, target_len, deadline):
        """
        Recursively try to extend the sequence to reach target_len.
        Returns True if a valid sequence of length target_len is found.
        """
        if time.time() > deadline:
            return False
        
        if len(sequence) == target_len:
            return True
        
        # Pruning: can we still reach target_len?
        # We need target_len - len(sequence) more elements
        # The smallest possible next element is last_val + 1
        # The largest possible next element is 256
        # So we have at most 256 - last_val candidates
        # But we need them to be strictly increasing, so the number of available
        # values is 256 - last_val
        # If 256 - last_val < target_len - len(sequence), we can't reach the target
        if 256 - last_val < target_len - len(sequence):
            return False
        
        # Try candidates from last_val + 1 to 256
        # Order: try smaller values first (they leave more room for future elements)
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return False
            
            # Compute new block sums when appending x
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(sequence):
                total += a
                new_sums.append(total)
            
            # Check for collisions
            collision = False
            for s in new_sums:
                if s in sums_set:
                    collision = True
                    break
            
            if collision:
                continue
            
            # Forward checking: after adding x, can we still reach target_len?
            # New last val is x, we need target_len - len(sequence) - 1 more elements
            if 256 - x < target_len - len(sequence) - 1:
                continue
            
            # Add x and recurse
            sequence.append(x)
            sums_set.update(new_sums)
            
            if _dfs(sequence, sums_set, x, target_len, deadline):
                return True
            
            # Backtrack
            sequence.pop()
            sums_set.difference_update(new_sums)
        
        return False
    
    # Try to find sequences longer than the greedy one
    # Start from target_len = best_len + 1, then best_len + 2, etc.
    # For each target length, try to find a valid sequence
    
    # We'll try different starting points
    # Start with empty sequence and try to build up to target_len
    
    # To speed up, we can use the greedy sequence as a hint
    # But for now, let's just do a full search
    
    # Try target lengths in increasing order
    for target_len in range(best_len + 1, best_len + 10):  # Try up to 10 more
        if time.time() > deadline:
            break
        
        # Reset and try to find a sequence of this length
        sequence = []
        sums_set = set()
        
        if _dfs(sequence, sums_set, 0, target_len, deadline):
            best_seq = list(sequence)
            best_len = target_len
            # Don't break; try to go even longer
        else:
            # If we can't find a sequence of this length, 
            # it might be possible that a different branch works, 
            # but given the time limit, we'll stop trying longer sequences
            # Actually, we should continue trying, as the search might just have timed out
            # But for efficiency, let's stop if we failed on a shorter length
            # unless we have time to spare
            if time.time() > deadline - 5:
                break
    
    return best_seq


def _greedy_construct():
    """Greedy construction: add the smallest possible value that doesn't create a sum collision."""
    values = []
    sums = set()
    
    for x in range(1, 257):
        # Compute new block sums if we append x
        new_sums = []
        total = x
        new_sums.append(total)
        for a in reversed(values):
            total += a
            new_sums.append(total)
        
        # Check for collisions
        if not any(s in sums for s in new_sums):
            values.append(x)
            sums.update(new_sums)
    
    return values

# EVOLVE-BLOCK-END
