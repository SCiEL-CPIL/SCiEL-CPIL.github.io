import sys
import numpy as np
import random

N = 18
NUM_INPUTS = 1 << N

def construct():
    """
    Constructs a sorting network for N=18 wires using Batcher's Merge Sort 
    followed by randomized greedy reduction.
    """
    
    def batcher_merge_sort(n):
        """Returns a list of comparators for sorting n wires using Batcher's Merge Sort."""
        if n <= 1:
            return []
        
        k = n // 2
        left = batcher_merge_sort(k)
        right = [(i + k, j + k) for (i, j) in batcher_merge_sort(n - k)]
        merge = batcher_odd_merge(k, n - k)
        
        return left + right + merge
    
    def batcher_odd_merge(n, m):
        """
        Returns comparators to merge two sorted lists of sizes n and m.
        The first list occupies indices 0..n-1, the second occupies indices n..n+m-1.
        Uses Batcher's Odd-Merge algorithm.
        """
        if n == 0 or m == 0:
            return []
        
        if n == 1 and m == 1:
            return [(0, 1)]
        
        # Use the standard Batcher Odd-Merge algorithm
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        # Merge A_odd (size n_odd) with B_even (size m_even)
        merge_odd = batcher_odd_merge(n_odd, m_even)
        
        def map_odd(pos):
            if pos < n_odd:
                return 2 * pos + 1
            else:
                return n + 2 * (pos - n_odd)
        
        mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
        
        # Merge A_even (size n_even) with B_odd (size m_odd)
        merge_even = batcher_odd_merge(n_even, m_odd)
        
        def map_even(pos):
            if pos < n_even:
                return 2 * pos
            else:
                return n + 2 * (pos - n_even) + 1
        
        mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
        
        # Final comparators
        total = n + m
        final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
        
        return mapped_odd + mapped_even + final_comps

    def is_sorted_network(network):
        """
        Checks if the network sorts all boolean inputs.
        Uses NumPy for vectorized simulation.
        """
        inputs = np.arange(NUM_INPUTS, dtype=np.uint8)
        wires = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
        for i in range(N):
            wires[i] = (inputs >> i) & 1
        
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        
        sorted_flags = np.all(wires[:-1] <= wires[1:], axis=0)
        return np.all(sorted_flags)

    def greedy_reduce(network):
        """
        Attempts to reduce the number of comparators by removing redundant ones.
        """
        current_network = list(network)
        improved = True
        while improved:
            improved = False
            for idx in range(len(current_network) - 1, -1, -1):
                candidate = current_network[:idx] + current_network[idx+1:]
                if is_sorted_network(candidate):
                    current_network = candidate
                    improved = True
                    break
        return current_network

    def randomized_greedy_reduce(network, num_restarts=3):
        """
        Performs multiple randomized greedy reductions.
        """
        best_network = network
        for _ in range(num_restarts):
            shuffled = list(network)
            random.shuffle(shuffled)
            reduced = greedy_reduce(shuffled)
            if len(reduced) < len(best_network):
                best_network = reduced
        return best_network

    seed_network = batcher_merge_sort(N)
    
    if not is_sorted_network(seed_network):
        return seed_network

    best = randomized_greedy_reduce(seed_network, num_restarts=3)
    
    return best

# EVOLVE-BLOCK-END
