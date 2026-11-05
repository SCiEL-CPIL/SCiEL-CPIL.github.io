"""
Constructs a 12-state binary deterministic automaton with the maximum
possible shortest reset word length (threshold).

The theoretical upper bound for the threshold of an N-state synchronizing
automaton is (N-1)^2 (Černý's conjecture, proven for N=12 and others).
For N=12, the bound is 11^2 = 121.

The "Černý automaton" construction achieves this bound.
States: 0, 1, ..., N-1
Letter a:
  0 -> 1
  i -> i+1 for 1 <= i <= N-2
  N-1 -> 1
Letter b:
  0 -> 0
  1 -> 0
  i -> i for 2 <= i <= N-1
"""
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard (trivial for this O(N) construction, but included per directive)
    start_time = time.time()
    deadline = start_time + 10.0  # 10s safety margin

    # Letter a: Cycle on 1..N-1, 0 enters at 1
    # 0 -> 1
    # 1 -> 2
    # ...
    # N-2 -> N-1
    # N-1 -> 1
    trans_a = [0] * N
    trans_a[0] = 1
    for i in range(1, N - 1):
        trans_a[i] = i + 1
    trans_a[N - 1] = 1

    # Letter b: Identity on 2..N-1, 0->0, 1->0
    # 0 -> 0
    # 1 -> 0
    # i -> i for i >= 2
    trans_b = [0] * N
    trans_b[0] = 0
    trans_b[1] = 0
    for i in range(2, N):
        trans_b[i] = i

    # Check deadline (should never trigger)
    if time.time() > deadline:
        # Fallback to a simple synchronizing automaton if somehow slow (impossible here)
        return [[(i + 1) % N for i in range(N)], [0] * N]

    return [trans_a, trans_b]
# EVOLVE-BLOCK-END
