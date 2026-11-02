# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 120s limit

    total = q ** n
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Build ball_matrix: ball_matrix[i, j] = 1 iff word j within `radius` of word i
    ball_matrix = _build_ball_matrix(words, q, n, radius)

    # Greedy set cover with random restarts
    best_code = None
    best_size = total + 1
    rng = np.random.default_rng(12345)

    num_restarts = 300
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        centers = _greedy_cover(ball_matrix, total, rng, restart)
        if len(centers) < best_size:
            best_size = len(centers)
            best_code = centers

    if best_code is None:
        return [list(w) for w in words]

    # Local search: remove redundant centers
    if time.time() < deadline - 3:
        best_code = _local_search(best_code, ball_matrix, total, deadline)
        best_size = len(best_code)

    return [list(words[i]) for i in best_code]


def _build_ball_matrix(words, q, n, radius):
    total = len(words)
    # For radius 1: word i covers j if they differ in at most 1 position
    # Compute pairwise distance efficiently
    # diff[i, j, pos] = words[i,pos] != words[j,pos]
    # Use broadcasting: words[:, None, :] vs words[None, :, :]
    # That's total x total x n which for 729^2*6 = ~3.2M, fine
    w1 = words[:, None, :]  # (total, 1, n)
    w2 = words[None, :, :]  # (1, total, n)
    diff = (w1 != w2)  # (total, total, n)
    num_diff = diff.sum(axis=2)  # (total, total)
    ball_matrix = (num_diff <= radius).astype(np.uint8)
    return ball_matrix


def _greedy_cover(ball_matrix, total, rng, restart):
    uncovered = np.ones(total, dtype=np.uint8)
    centers = []
    # Random perturbation to break ties and explore different paths
    noise_scale = 0.5 + 0.3 * (restart % 5)
    noise = rng.random(total) * noise_scale
    while uncovered.any():
        gains = ball_matrix @ uncovered  # (total,)
        gains = gains + noise
        i = int(np.argmax(gains))
        cov = ball_matrix[i]
        if (cov & uncovered).sum() == 0:
            # fallback without noise
            gains2 = ball_matrix @ uncovered
            i = int(np.argmax(gains2))
            if (ball_matrix[i] & uncovered).sum() == 0:
                break
        centers.append(i)
        uncovered[ball_matrix[i]] = 0
        # Occasionally refresh noise to change trajectory
        if len(centers) % 10 == 0:
            noise = rng.random(total) * noise_scale
    return centers


def _local_search(code, ball_matrix, total, deadline):
    code = list(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Precompute coverage of each center
        for i in range(len(code)):
            rest = code[:i] + code[i+1:]
            if not rest:
                continue
            # Vectorized coverage check
            covered = ball_matrix[rest].sum(axis=0)
            if (covered > 0).all():
                code = rest
                improved = True
                break
    return code
# EVOLVE-BLOCK-END
