"""Rewritten program using MODEL-AND-OPTIMIZE approach with integer-bit construction."""
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
    
    Construction strategy: Represent the solution as a single integer where all 
    N lower bits are set to 1 (i.e., value = 2^N - 1), then extract the binary 
    representation formatted to exactly N bits. This leverages arbitrary-precision 
    integer arithmetic as a distinct computational pathway from string repetition.
    
    Final polish: Verify the constructed string has exactly N ones and is valid.
    """
    start_time = time.time()
    
    # Check deadline before starting heavy computation
    if time.time() - start_time > DEADLINE_LIMIT:
        # Fallback: return all ones string directly
        return '1' * N
    
    # Construct the optimal integer: all N bits set to 1
    # This is equivalent to 2^N - 1
    # Using bitwise operations: start with 0, shift left N times and OR with 1 each time
    # But more efficiently: (1 << N) - 1 gives us N ones in binary
    
    # Optimization: Use the direct formula for all-ones integer
    optimal_int = (1 << N) - 1
    
    # Verify we haven't exceeded deadline
    if time.time() - start_time > DEADLINE_LIMIT:
        return '1' * N
    
    # Extract the lower N bits as a binary string
    # format(optimal_int, '0100b') gives exactly 100-bit binary representation
    # Since optimal_int = 2^100 - 1, its binary representation is exactly 100 ones
    result = format(optimal_int, f'0{N}b')
    
    # Final high-precision polish: verify the result is valid and optimal
    # Check that we have exactly N bits
    if len(result) != N:
        # Truncate or pad if necessary
        if len(result) > N:
            result = result[:N]
        else:
            result = result.rjust(N, '1')
    
    # Verify all bits are '1' (optimal for OneMax)
    # Count the ones to confirm optimality
    if time.time() - start_time <= DEADLINE_LIMIT:
        ones_count = result.count('1')
        if ones_count != N:
            # This should never happen with our construction, but as a safety net
            result = '1' * N
    
    return result
# EVOLVE-BLOCK-END
