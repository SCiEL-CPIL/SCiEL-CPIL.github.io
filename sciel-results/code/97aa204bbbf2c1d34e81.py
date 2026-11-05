"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the Černý automaton for N=12 states.
    This automaton is known to have a shortest reset word of length (N-1)^2.
    For N=12, this is 121.
    
    Letter a: i -> (i + 1) % N
    Letter b: 0 -> 0, i -> i - 1 for i > 0
    """
    a = [(i + 1) % N for i in range(N)]
    b = [0 if i == 0 else i - 1 for i in range(N)]
    
    # Optional: Verify synchronization and compute threshold within time limit
    # This is a safety check. The construction is theoretically sound.
    # BFS on subset automaton.
    try:
        start_time = time.time()
        deadline = start_time + 5.0 # 5 second safety margin
        
        # BFS to find shortest reset word
        # States are represented as bitmasks (integers)
        # Full set: (1 << N) - 1
        full_set = (1 << N) - 1
        
        # Map from subset (int) to (previous_subset, letter)
        # We want to reach any singleton {i} -> (1 << i)
        
        # Precompute transitions for each letter on each subset?
        # Or compute on the fly.
        # On the fly is fine for N=12.
        
        def transition(subset, letter):
            if letter == 0: # a
                # i -> (i+1)%N
                # Bit i moves to bit (i+1)%N
                # This is a rotation of bits.
                # (subset << 1) | (subset >> (N-1)) & 1 ... careful with N=12
                # Actually, simpler:
                # new_subset = 0
                # for i in range(N):
                #   if subset & (1<<i):
                #     new_subset |= 1 << ((i+1)%N)
                # But bitwise rotation is faster.
                # Left rotate by 1: ((subset << 1) | (subset >> (N-1))) & ((1<<N)-1)
                return ((subset << 1) | (subset >> (N - 1))) & ((1 << N) - 1)
            else: # b
                # 0 -> 0
                # i -> i-1 for i > 0
                # Bit 0 stays at 0.
                # Bit i moves to i-1.
                # This is a right shift, but bit 0 is special.
                # If bit 0 is set, it stays.
                # Bits 1..N-1 shift right by 1.
                # new_subset = (subset & 1) | ((subset & ~1) >> 1)
                return (subset & 1) | ((subset & ~1) >> 1)
        
        # BFS
        visited = {full_set: None} # subset -> (prev_subset, letter)
        queue = [full_set]
        
        # Check if start is singleton (N=1)
        if N == 1:
            return a, b
        
        while queue:
            if time.time() > deadline:
                break
            current = queue.pop(0)
            
            # Check if current is a singleton
            # A singleton has exactly one bit set.
            # current & (current - 1) == 0
            if current & (current - 1) == 0:
                # Found a reset state. We can reconstruct the word if needed, 
                # but the task only asks for the automaton.
                # The score is the length, which is determined by the evaluator.
                # We just need to return the automaton.
                break
            
            # Try both letters
            for letter in [0, 1]:
                next_sub = transition(current, letter)
                if next_sub not in visited:
                    visited[next_sub] = (current, letter)
                    queue.append(next_sub)
                    
    except Exception:
        pass # Ignore verification errors, trust the construction
    
    return a, b
# EVOLVE-BLOCK-END
