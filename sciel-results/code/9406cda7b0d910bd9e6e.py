"""Rewritten program using MODEL-AND-OPTIMIZE approach."""
import time

N = 100
DEADLINE_LIMIT = 145  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Model-and-Optimize approach for OneMax (100 bits).
    
    Model: Each bit $i$ is generated with probability $p_i$ of being '1'.
    Objective: Maximize the expected number of ones, or in the deterministic 
    construction case, maximize the actual count of ones.
    
    Optimization:
    Since we can construct the string directly, we can choose the bit values 
    to maximize the objective function $J = \sum_{i=1}^N b_i$, where $b_i \in \{0, 1\}$.
    This is equivalent to optimizing the parameters $p_i$ in a Bernoulli model 
    where we can set $p_i = 1$ to guarantee $b_i = 1$.
    
    The optimal parameters are $p_i = 1.0$ for all $i$.
    
    Final polish: Verify the constructed string has exactly N ones and is valid.
    """
    start_time = time.time()
    
    # Define the model parameters: probability of '1' for each bit
    # Initialize with a baseline
    p = [0.5] * N
    
    # Optimize parameters against the objective
    # The objective for a deterministic string of bits b is sum(b).
    # To maximize sum(b) where b_i is derived from p_i, we set p_i to the 
    # value that maximizes the contribution. For a single trial construction, 
    # setting p_i = 1.0 ensures b_i = 1, contributing 1 to the sum.
    # Any p_i < 1.0 has a non-zero chance (or expectation) of being 0, 
    # which is suboptimal.
    
    # Direct optimization step: Set all probabilities to 1.0
    # This is the global optimum for maximizing the count of ones.
    for i in range(N):
        if time.time() - start_time > DEADLINE_LIMIT:
            break
        p[i] = 1.0
        
    # Construct the binary string based on the optimized parameters
    # Since p[i] is 1.0, the bit is deterministically '1'
    bits = []
    for i in range(N):
        if time.time() - start_time > DEADLINE_LIMIT:
            # If we hit the deadline mid-construction, return what we have 
            # padded with '1's to maintain validity
            break
        if p[i] == 1.0:
            bits.append('1')
        else:
            bits.append('0')
            
    # If we didn't complete the loop due to time, pad with '1's to ensure
    # we have a valid length-N string with maximum possible ones
    if len(bits) < N:
        bits.extend(['1'] * (N - len(bits)))
    
    # Final high-precision polish: verify the result is valid and optimal
    # Check that we have exactly N bits
    if len(bits) != N:
        # Truncate or pad if necessary
        if len(bits) > N:
            bits = bits[:N]
        else:
            bits.extend(['1'] * (N - len(bits)))
    
    # Verify all bits are valid ('0' or '1')
    # For OneMax, we want all ones, so this is a sanity check
    result = "".join(bits)
    
    # Optional: Verify the count of ones is N (optimal)
    # This is a final polish step to ensure we achieved the maximum
    # If somehow we have fewer than N ones, we could try to fix it,
    # but given our construction, this should always be N.
    
    return result
# EVOLVE-BLOCK-END
