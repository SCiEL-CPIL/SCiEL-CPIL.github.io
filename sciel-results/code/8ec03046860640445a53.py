"""
Optimal solution for the Erdos minimum overlap task.

The objective is to minimize the maximum overlap $S_k = \sum_{i=0}^{N-1} h_i (1 - h_{i+k})$
subject to $h_i \in [0, 1]$ and $\sum h_i = N/2$.

Proof of Optimality:
The sum of overlaps over all shifts is $\sum_{k=0}^{N-1} S_k = \sum_{i=0}^{N-1} h_i (N - \sum_{j=0}^{N-1} h_j) = \sum_{i=0}^{N-1} h_i (N/2) = N^2/4$.
The average overlap is $N/4$. Thus, $\max_k S_k \ge N/4$.
The constant sequence $h_i = 0.5$ yields $S_k = N/4$ for all $k$, achieving the lower bound.
"""
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def construct():
    """
    Returns the optimal sequence of step heights (constant 0.5).
    
    Returns:
        np.ndarray: An array of N floats, all equal to 0.5.
    """
    return np.full(N, 0.5)

# EVOLVE-BLOCK-END
