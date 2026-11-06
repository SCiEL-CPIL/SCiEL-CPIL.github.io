"""Two Sidon sets with disjoint difference sets – Fast incremental search with local improvement."""

import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 55.0  # Safety margin before 90s limit (evaluator kills at 90s, but let's be safe)


class SidonState:
    """Incremental state for checking Sidon property and difference disjointness."""
    
    def __init__(self, elements):
        self.elements = list(elements)
        self.sums = set()
        self.diffs = set()
        self._rebuild()
    
    def _rebuild(self):
        self.sums = set()
        self.diffs = set()
        for i, a in enumerate(self.elements):
            for b in self.elements[i:]:
                self.sums.add(a + b)
            for b in self.elements:
                d = b - a
                if d != 0:
                    self.diffs.add(d)
    
    def can_add(self, x, other_diffs):
        """Check if x can be added to this set."""
        # Check sum collisions
        for a in self.elements:
            if (a + x) in self.sums:
                return False
        if (x + x) in self.sums:
            return False
        # Check difference collisions with other set
        for a in self.elements:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
        return True
    
    def add(self, x):
        """Add x to the set, updating sums and diffs."""
        self.elements.append(x)
        for a in self.elements[:-1]:  # all previous elements
            self.sums.add(a + x)
            d1 = x - a
            d2 = a - x
            self.diffs.add(d1)
            self.diffs.add(d2)
        self.sums.add(x + x)


def greedy_build(order_a, order_b, deadline):
    """Build two Sidon sets greedily following given element orders."""
    state_a = SidonState([])
    state_b = SidonState([])
    
    for x in order_a:
        if time.time() > deadline:
            break
        if len(state_a.elements) >= MAX_SIZE:
            break
        if state_a.can_add(x, state_b.diffs):
            state_a.add(x)
    
    for x in order_b:
        if time.time() > deadline:
            break
        if len(state_b.elements) >= MAX_SIZE:
            break
        if state_b.can_add(x, state_a.diffs):
            state_b.add(x)
    
    return state_a.elements, state_b.elements


def local_improve(sets_a, sets_b, deadline):
    """Try to improve by swapping or adding elements."""
    a = list(sets_a)
    b = list(sets_b)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add new elements to A
        for x in range(1, N + 1):
            if x in a:
                continue
            state_a = SidonState(a)
            state_b = SidonState(b)
            if state_a.can_add(x, state_b.diffs):
                a.append(x)
                improved = True
                break
        # Try to add new elements to B
        if not improved:
            for x in range(1, N + 1):
                if x in b:
                    continue
                state_a = SidonState(a)
                state_b = SidonState(b)
                if state_b.can_add(x, state_a.diffs):
                    b.append(x)
                    improved = True
                    break
    return a, b


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_a, best_b = [], []
    best_size = 0
    
    # Multiple restarts with different random orders
    random.seed(42)
    
    while time.time() < deadline - 2.0:  # Leave 2s margin
        # Create random permutations
        order_a = list(range(1, N + 1))
        random.shuffle(order_a)
        order_b = list(range(1, N + 1))
        random.shuffle(order_b)
        
        a, b = greedy_build(order_a, order_b, deadline - 1.0)
        current_size = len(a) + len(b)
        
        if current_size > best_size:
            best_size = current_size
            best_a, best_b = a, b
        
        # Try local improvement if we have time
        if time.time() < deadline - 3.0 and current_size >= best_size:
            a_imp, b_imp = local_improve(a, b, deadline - 1.0)
            imp_size = len(a_imp) + len(b_imp)
            if imp_size > best_size:
                best_size = imp_size
                best_a, best_b = a_imp, b_imp
    
    # Ensure nonempty
    if not best_a:
        best_a = [1]
    if not best_b:
        best_b = [2]
    
    return [best_a, best_b]
