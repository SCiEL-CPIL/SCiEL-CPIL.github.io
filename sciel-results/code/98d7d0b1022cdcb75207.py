# EVOLVE-BLOCK-START
import numpy as np


def _local_density(prizes, dist, max_len):
    """Constraint-aware marginal utility.

    For each node j, reward = sum over nearby nodes i (within a fraction of
    the budget) of the prize you could collect by routing i -> j -> 0, i.e.
    prize_j * prize_i * exp(-c * (d_ij + d_j0)). This is a smooth surrogate
    for 'how much value j unlocks given the remaining tour budget', so it
    concentrates the heuristic on nodes that sit inside dense, reachable
    prize clusters rather than just being individually valuable.
    """
    # reachability of j given we must still return to 0
    reach_j = np.exp(-(dist + dist[:, 0]) / max_len)          # (n, n)
    # value that j unlocks: pairwise prize product gated by reachability
    w = prizes[None, :] * prizes[:, None] * reach_j
    dens = w.sum(axis=1)                                       # (n,)
    return dens + 1e-9


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    dist = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    # base marginal utility: prize of destination / round-trip cost
    rt = dist + dist[:, 0]                       # cost i -> j -> 0
    rt = np.where(rt > 0, rt, 1.0)
    base = prizes[None, :] / (rt ** 0.7)

    # constraint-aware local density (smooth surrogate)
    dens = _local_density(prizes, dist, max_len)

    # combine: node-specific density factor modulates the edge base
    h = base * dens[None, :]

    # keep it finite, positive, and diagonal-safe
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)

    return h
# EVOLVE-BLOCK-END
