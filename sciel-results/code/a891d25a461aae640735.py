"""Construct a pentagon-free graph with no independent 9-set, maximizing n.

Design: 8 disjoint copies of K4.
- Each K4 is a clique of size 4, so it contains no 5-cycle (a 5-cycle needs 5 distinct
  vertices; every component here has only 4).
- The independence number of a disjoint union is the sum of the independence numbers
  of the components. Each K4 has independence number 1, so the whole graph has
  alpha = 8 < 9.
- n = 8 * 4 = 32, which is the best known valid construction for R(C5, K9) > 32.
"""
# EVOLVE-BLOCK-START
import time

def construct():
    # Hard wall-clock safety margin well inside the 120s evaluator limit.
    deadline = time.time() + 10.0  # this construction is O(1); margin is precautionary

    # 8 disjoint copies of K4 on vertices 0..31.
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Guard: never return past the deadline (construction is instant, but keep the
    # guard per the compute-budget contract).
    if time.time() > deadline:
        # Fallback: still valid (7 disjoint K4s, n=28, alpha=7<9, no 5-cycle).
        n = 28
        edges = [[4 * b + i, 4 * b + j] for b in range(7)
                 for i in range(4) for j in range(i + 1, 4)]

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
