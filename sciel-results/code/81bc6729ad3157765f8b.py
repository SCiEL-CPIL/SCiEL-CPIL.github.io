"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import math

def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using the zero-one principle.
    Represents each wire's state across all 65536 inputs as a Python int bitmask.
    """
    n = N
    # Initialize wire states: wire w for input k is bit w of k
    wire_state = []
    for w in range(n):
        # Build bitmask where bit k = (k >> w) & 1 for k in 0..2^n-1
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    # Apply each comparator
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b  # min for 0/1
        wire_state[j] = a | b  # max for 0/1
    
    # Check sorted: wire_state[k] & ~wire_state[k+1] must be 0 for all k
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """
    Generate a Bitonic Sorting Network for n wires.
    n must be a power of 2.
    """
    network = []
    
    def bitonic_sort_recursive(size, offset):
        """
        Sorts the sub-array of length 'size' starting at 'offset' in ascending order.
        size must be a power of 2.
        """
        if size == 2:
            network.append((offset, offset + 1))
            return
        
        half = size // 2
        # Sort first half ascending
        bitonic_sort_recursive(half, offset)
        # Sort second half ascending
        bitonic_sort_recursive(half, offset + half)
        
        # Merge the two sorted halves into a single sorted half.
        # The merge step for bitonic sort of size 'size' involves:
        # Comparing elements at distance 'half'.
        # Then recursively merging the two resulting 'half'-sized sequences.
        
        def merge_recursive(m_size, m_offset):
            if m_size == 2:
                network.append((m_offset, m_offset + 1))
                return
            
            m_half = m_size // 2
            # Compare elements at distance m_half
            for i in range(m_half):
                network.append((m_offset + i, m_offset + i + m_half))
            
            # Recursively merge the two halves
            merge_recursive(m_half, m_offset)
            merge_recursive(m_half, m_offset + m_half)

        merge_recursive(size, offset)

    bitonic_sort_recursive(N, 0)
    return network

def construct():
    start_time = time.time()
    # The evaluator kills at 90s. We use a 50s deadline to be safe, 
    # but we can push it a bit higher if the verification is fast enough.
    # Let's stick to 50s to ensure we return within 60s as per task contract 
    # "Each evaluation must finish within 60 s".
    deadline = start_time + 50.0
    
    # Start with Bitonic Network
    network = _generate_bitonic_network(N)
    
    if not _verify_network_fast(network):
        # Fallback to odd-even if bitonic generation failed (should not happen)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    best_network = network
    best_count = len(network)
    
    # Local Search: Prune redundant comparators
    # We iterate through the network, trying to remove each comparator.
    # If removal preserves correctness, we keep it removed.
    # We repeat this process until no more comparators can be removed.
    
    # To make it faster, we can randomize the order of checks.
    # Also, we can use a "simulated annealing" like approach if we get stuck, 
    # but simple pruning from a good start is often effective.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(best_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            candidate = best_network[:idx] + best_network[idx+1:]
            
            # Optimization: If the network is empty, it's only valid if N=1
            if _verify_network_fast(candidate):
                best_network = candidate
                improved = True
                break # Restart scan from beginning (or random) after change
        
        # If we improved, we might want to do another pass.
        # The loop condition handles this.

    # If we have time, try to add random comparators and prune again to escape local minima?
    # Given the strong start (60 comparators), and the known optimum (54), 
    # simple pruning might get us close.
    
    # Let's do a few restarts with perturbation if time permits.
    if time.time() < deadline - 5.0:
        for _ in range(5): # 5 restarts
            if time.time() > deadline:
                break
            
            current = best_network[:]
            # Add 2 random comparators
            for _ in range(2):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(current))
                current.insert(pos, (i, j))
            
            # Prune
            improved = True
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(len(current)))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    candidate = current[:idx] + current[idx+1:]
                    if _verify_network_fast(candidate):
                        current = candidate
                        improved = True
                        break
            
            if len(current) < best_count:
                best_network = current
                best_count = len(current)

    return best_network
# EVOLVE-BLOCK-END
