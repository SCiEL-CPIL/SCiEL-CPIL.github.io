import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _verify_network(nets):
    """Verify network sorts all 2^N binary inputs using 0-1 principle."""
    total = 1 << N
    bits = np.arange(total, dtype=np.uint8)
    wires = np.empty((total, N), dtype=np.uint8)
    for k in range(N):
        wires[:, k] = (bits >> k) & 1
    for (i, j) in nets:
        wi = wires[:, i].copy()
        wj = wires[:, j].copy()
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    for k in range(N - 1):
        if np.any(wires[:, k] > wires[:, k+1]):
            return False
    return True

def construct():
    deadline = time.time() + 85
    
    # Start with odd-even transposition network
    nets = [(i, i+1) for phase in range(N) for i in range(phase % 2, N-1, 2)]
    
    # Verify base
    if not _verify_network(nets):
        return nets
    
    # Aggressive greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(nets) and time.time() < deadline:
            new_nets = nets[:i] + nets[i+1:]
            if _verify_network(new_nets):
                nets = new_nets
                improved = True
            else:
                i += 1
    
    return nets

# EVOLVE-BLOCK-END
