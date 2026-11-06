"""Snake-in-the-box in Q_9: construct the longest induced path (snake)."""
import time
import numpy as np

N = 9
SIZE = 1 << N
SEED_LEN = 104  # known length of the naive greedy snake in Q_9
DEADLINE = 120.0


def _valid_mask(path, used_mask):
    """For a path (list of ints), return a bool array of candidate next vertices.

    A candidate v is valid iff:
      - not used,
      - dist(cur, v) == 1 (adjacent to the tail), and
      - dist(v, p) >= 2 for every earlier vertex p (keeps the path induced).
    """
    cur = path[-1]
    # vertices at distance exactly 1 from the tail
    adj = np.zeros(SIZE, dtype=bool)
    for b in range(N):
        adj[cur ^ (1 << b)] = True
    # forbid all vertices at distance 1 from any earlier vertex
    for p in path[:-1]:
        for b in range(N):
            adj[p ^ (1 << b)] = False
    adj &= ~used_mask
    return adj


def _greedy(path, used_mask, mask, deadline):
    """Greedily extend the path using the current valid mask; return the extended path."""
    i = 0
    while i < len(path):
        cur = path[-1]
        cand = np.nonzero(mask)[0]
        nxt = None
        for v in cand:
            iv = int(v)
            if (iv ^ cur).bit_count() == 1 and (iv not in used_mask):
                nxt = iv
                break
        if nxt is None:
            break
        path.append(nxt)
        used_mask[nxt] = True
        if time.time() > deadline:
            break
        mask = _valid_mask(path, used_mask)
        i += 1
    return path


def _lookahead2(path, used_mask, mask, deadline):
    """2-ply lookahead: for each valid first step, take the best greedy continuation."""
    cur = path[-1]
    firsts = [int(v) for v in np.nonzero(mask)[0]
              if (int(v) ^ cur).bit_count() == 1 and int(v) not in used_mask]
    best = None
    best_len = len(path)
    for v in firsts:
        p2 = path + [v]
        m2 = _valid_mask(p2, used_mask)
        p3 = _greedy(p2, used_mask, m2, deadline)
        if len(p3) > best_len:
            best_len = len(p3)
            best = p3
        if time.time() > deadline:
            break
    return best if best is not None else path


def construct():
    deadline = time.time() + DEADLINE - 5.0  # safety margin

    # Initial path: 0, then 1, then greedy.
    used_mask = np.zeros(SIZE, dtype=bool)
    used_mask[0] = True
    path = [0, 1]
    used_mask[1] = True
    mask = _valid_mask(path, used_mask)
    path = _greedy(path, used_mask, mask, deadline)

    # Repeatedly improve the tail via 2-ply lookahead while time permits.
    while time.time() < deadline:
        mask = _valid_mask(path, used_mask)
        new_path = _lookahead2(path, used_mask, mask, deadline)
        if new_path is None or len(new_path) <= len(path):
            break
        path = new_path
        for v in path:
            used_mask[v] = True

    return path
