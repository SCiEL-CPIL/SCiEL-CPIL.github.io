"""Pruning local search for a minimal 16-wire sorting network."""
import time
import random

N = 16
ZERO_ONE_TOTAL = 1 << N  # 65536


def _verify(network):
    """Return True iff the compare-exchange network sorts all 2^16 Boolean inputs."""
    for mask in range(ZERO_ONE_TOTAL):
        vals = [(mask >> i) & 1 for i in range(N)]
        for i, j in network:
            a = vals[i]
            b = vals[j]
            if a > b:
                vals[i] = b
                vals[j] = a
        for i in range(1, N):
            if vals[i - 1] > vals[i]:
                return False
    return True


def _seed_network():
    """Odd-even transposition sorting network (guaranteed valid)."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin under the 90s kill
    rng = random.Random(12345)

    best = _seed_network()
    best_len = len(best)
    assert _verify(best)  # sanity: seed must be valid

    # Candidate comparators: all distinct pairs, ordered deterministically.
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(all_pairs)

    while time.time() < deadline:
        # Greedy forward pruning: drop each comparator if the network still sorts.
        pruned = []
        for comp in best:
            trial = pruned + [comp]
            if _verify(trial):
                pruned.append(comp)
        if len(pruned) < best_len:
            best = pruned
            best_len = len(best)
            if best_len <= 60:  # cannot beat the 60-comparator reference
                break

        # Try a few random restarts to escape local minima.
        for _ in range(3):
            if time.time() >= deadline:
                break
            cur = list(best)
            improved = True
            while improved:
                improved = False
                # Randomly insert a candidate comparator, then re-prune.
                for comp in rng.sample(all_pairs, min(40, len(all_pairs))):
                    if time.time() >= deadline:
                        break
                    if comp in cur:
                        continue
                    trial = cur + [comp]
                    if _verify(trial):
                        cur = trial
                        improved = True
                # Prune the enlarged network.
                pruned = []
                for comp in cur:
                    trial = pruned + [comp]
                    if _verify(trial):
                        pruned.append(comp)
                if len(pruned) < len(cur):
                    cur = pruned
                    improved = True
            if len(cur) < best_len:
                best = cur
                best_len = len(cur)
                if best_len <= 60:
                    break
            if best_len <= 60:
                break

    return best
# EVOLVE-BLOCK-END
