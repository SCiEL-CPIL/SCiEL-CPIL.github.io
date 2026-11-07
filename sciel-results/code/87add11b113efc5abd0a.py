"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    # prize[j]: marginal gain of visiting node j (row of the matrix: from i to j)
    prize_col = prizes[None, :]                      # shape (1, n)

    # distance cost i -> j
    dij = D                                          # shape (n, n)

    # remaining budget slack after going i->j and then j->0
    # feasible only if this is > 0 (roughly, ignoring length-so-far)
    slack = max_len - dij - D[:, 0][None, :]         # shape (n, n)

    # prize density: prize[j] / distance i->j
    density = prize_col / (dij + 1e-9)               # shape (n, n)

    # weight by how much slack remains (more slack = more attractive)
    slack_w = np.clip(slack, 0.0, None)              # infeasible -> 0
    # normalize slack to [0,1] per column for stable weighting
    slack_max = np.max(slack_w, axis=0, keepdims=True)
    slack_max = np.where(slack_max > 0, slack_max, 1.0)
    slack_norm = slack_w / slack_max                  # shape (n, n)

    # combined: high prize density * comfortable budget slack
    H = density * (0.5 + slack_norm)                 # base 0.5 keeps feasible edges active

    # strongly suppress infeasible edges (slack <= 0)
    H = np.where(slack_w > 0, H, 1e-9)

    # diagonal (self loops) should not be attractive
    H = np.where(np.eye(n, dtype=bool), 1e-9, H)

    # ensure finiteness and positivity
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)

    return H
# EVOLVE-BLOCK-END
