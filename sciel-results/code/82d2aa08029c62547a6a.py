"""Optimized pancyclic graph construction on 40 vertices using cycle + chords."""
N = 40
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s limit

    # Base cycle: 0-1-2-...-39-0 (40 edges)
    edges = [(i, (i + 1) % N) for i in range(N)]

    # Add chords (0, k) for k = 2..20 (19 edges)
    # Each chord (0,k) creates cycles of length k+1 and 41-k
    # Together they cover all lengths 3..39
    for k in range(2, 21):
        edges.append((0, k))

    # Generate cycles of every length 3..40
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            k = length - 1
            if 2 <= k <= 20:
                # Cycle 0-1-2-...-k-0 (length k+1)
                cyc = list(range(k + 1))
            else:
                # Cycle 0-39-38-...-k-0 (length 41-k)
                k = 41 - length
                cyc = [0] + list(range(39, k - 1, -1))
            cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
