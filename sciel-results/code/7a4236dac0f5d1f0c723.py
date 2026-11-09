"""Rewritten program for the Černý problem (N=12).

Strategy:
- Start from the Černý automaton (cycle + a map that sends 0->0 and i->i-1).
- Use local search: random single-edge swaps (or letter reassignments) that
  preserve synchronizability, accepting moves that increase the exact shortest
  reset-word length (computed via BFS on the subset automaton).
- Hard deadline guard: stop before the 90s wall-clock limit and return the best
  automaton found.
"""
import time
import random
from collections import deque

N = 12

def _reset_word_length(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton.
    Returns None if the automaton is not synchronizing."""
    # subset represented as a bitmask of N bits
    full = (1 << N) - 1
    # map letter to transition function on a single state
    trans = [a, b]

    # BFS from the full set
    dist = {full: 0}
    q = deque([full])
    while q:
        s = q.popleft()
        d = dist[s]
        if s == 1:  # single state (bit 0)
            return d
        for letter in trans:
            # apply letter to each state in the subset
            ns = 0
            for i in range(N):
                if (s >> i) & 1:
                    ns |= (1 << letter[i])
            if ns not in dist:
                dist[ns] = d + 1
                q.append(ns)
    return None  # not synchronizing


def _cernet_automaton():
    """Černý automaton: a = cycle, b = (0->0, i->i-1)."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] + list(range(1, N))  # b[0]=0, b[i]=i-1 for i>=1
    return a, b


def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    rng = random.Random(12345)

    a, b = _cernet_automaton()
    best_len = _reset_word_length(a, b)
    best_a, best_b = list(a), list(b)

    if best_len is None:
        # fallback (should not happen)
        return best_a, best_b

    # Local search: try to increase the reset-word length.
    # A "move" picks a letter, a state, and a new destination.
    iters = 0
    while time.time() < deadline:
        iters += 1
        # Choose a random move: modify one edge in one letter
        letter_idx = rng.randrange(2)
        state = rng.randrange(N)
        cur = a if letter_idx == 0 else b
        new_dest = rng.randrange(N)
        if new_dest == cur[state]:
            continue
        # Apply
        old = cur[state]
        cur[state] = new_dest
        L = _reset_word_length(a, b)
        if L is not None and L > best_len:
            best_len = L
            best_a, best_b = list(a), list(b)
            # Occasionally, to escape local optima, keep going (no explicit SA
            # needed; pure hill-climbing on this small space is fine).
        else:
            # Revert
            cur[state] = old

        # Periodic restarts to escape local optima: with small probability,
        # jump to a random perturbation of the best found so far.
        if iters % 500 == 0 and rng.random() < 0.1:
            # Perturb best_a/best_b by a few random edge changes
            a, b = list(best_a), list(best_b)
            for _ in range(rng.randrange(1, 4)):
                li = rng.randrange(2)
                st = rng.randrange(N)
                nd = rng.randrange(N)
                (a if li == 0 else b)[st] = nd
            L = _reset_word_length(a, b)
            if L is None:
                # revert to best
                a, b = list(best_a), list(best_b)

    return best_a, best_b
