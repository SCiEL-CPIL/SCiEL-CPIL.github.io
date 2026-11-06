import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s limit

    n = 256

    def get_diffs(S):
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    class SetState:
        __slots__ = ['elements', 'sums', 'diffs']

        def __init__(self):
            self.elements = []
            self.sums = set()
            self.diffs = set()

        def try_add(self, x, other_diffs):
            if x in self.elements:
                return False, None, None
            new_sums = set()
            for s in self.elements:
                ns = x + s
                if ns in self.sums or ns in new_sums:
                    return False, None, None
                new_sums.add(ns)
            two_x = 2 * x
            if two_x in self.sums or two_x in new_sums:
                return False, None, None
            new_sums.add(two_x)

            new_diffs = set()
            for s in self.elements:
                d = abs(x - s)
                if d in other_diffs or d in self.diffs or d in new_diffs:
                    return False, None, None
                new_diffs.add(d)
            return True, new_sums, new_diffs

        def commit(self, x, new_sums, new_diffs):
            self.elements.append(x)
            self.sums.update(new_sums)
            self.diffs.update(new_diffs)

        def remove(self, x):
            self.elements.remove(x)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

        def rebuild(self, elements):
            self.elements = list(elements)
            self.sums = set()
            self.diffs = set()
            for i in range(len(self.elements)):
                for j in range(i, len(self.elements)):
                    self.sums.add(self.elements[i] + self.elements[j])
            self.diffs = get_diffs(self.elements)

    def greedy_with_balance(ordering, max_per_set=64):
        A = SetState()
        B = SetState()
        for x in ordering:
            if len(A.elements) < len(B.elements) or (len(A.elements) == len(B.elements) and len(A.elements) % 2 == 0):
                order = [(A, B), (B, A)]
            else:
                order = [(B, A), (A, B)]
                
            for target, other in order:
                if len(target.elements) >= max_per_set:
                    continue
                success, new_sums, new_diffs = target.try_add(x, other.diffs)
                if success:
                    target.commit(x, new_sums, new_diffs)
                    break
        return A, B

    def simulated_annealing(A_init, B_init, max_time=30.0):
        A = SetState()
        B = SetState()
        A.rebuild(A_init)
        B.rebuild(B_init)

        best_total = len(A.elements) + len(B.elements)
        best_A = list(A.elements)
        best_B = list(B.elements)

        all_elements = set(range(1, n + 1))
        
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        current_total = len(A.elements) + len(B.elements)
        
        start_sa = time.time()
        
        while time.time() < start_sa + max_time and T > min_T:
            used = set(A.elements) | set(B.elements)
            unused = list(all_elements - used)
            
            move_type = random.random()
            
            new_total = current_total
            move_successful = False
            undone = None
            
            if move_type < 0.4: # Add
                if unused:
                    x = random.choice(unused)
                    targets = [(A, B), (B, A)]
                    random.shuffle(targets)
                    for target, other in targets:
                        if len(target.elements) >= 64: continue
                        success, ns, nd = target.try_add(x, other.diffs)
                        if success:
                            target.commit(x, ns, nd)
                            new_total += 1
                            move_successful = True
                            undone = ('add', target, x)
                            break
            elif move_type < 0.7: # Remove
                src = A if random.random() < 0.5 else B
                if src.elements:
                    x = random.choice(src.elements)
                    src.remove(x)
                    new_total -= 1
                    move_successful = True
                    undone = ('remove', src, x)
            else: # Move A<->B
                src = A if random.random() < 0.5 else B
                dst = B if src == A else A
                if src.elements:
                    x = random.choice(src.elements)
                    src.remove(x)
                    success, ns, nd = dst.try_add(x, src.diffs)
                    if success:
                        dst.commit(x, ns, nd)
                        move_successful = True
                        undone = ('move', src, dst, x)
                    else:
                        src.rebuild(src.elements + [x])
                        move_successful = False

            if move_successful:
                delta = new_total - current_total
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_total = new_total
                    if current_total > best_total:
                        best_total = current_total
                        best_A = list(A.elements)
                        best_B = list(B.elements)
                else:
                    if undone[0] == 'add':
                        _, target, x = undone
                        target.remove(x)
                    elif undone[0] == 'remove':
                        _, src, x = undone
                        other = B if src == A else A
                        success, ns, nd = src.try_add(x, other.diffs)
                        if success:
                            src.commit(x, ns, nd)
                        else:
                            src.rebuild(src.elements + [x])
                    elif undone[0] == 'move':
                        _, src, dst, x = undone
                        dst.remove(x)
                        success, ns, nd = src.try_add(x, dst.diffs)
                        if success:
                            src.commit(x, ns, nd)
                        else:
                            src.rebuild(src.elements + [x])
            
            T *= cooling

        return best_A, best_B

    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]

    inter = []
    for i in range(1, n // 2 + 1):
        inter.append(i)
        inter.append(n + 1 - i)
    orderings.append(inter[:n])

    rng = random.Random(12345)
    for _ in range(10):
        s = list(range(1, n + 1))
        rng.shuffle(s)
        orderings.append(s)

    best_total = 0
    best_A = None
    best_B = None

    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        A_state, B_state = greedy_with_balance(ordering, max_per_set=64)
        total = len(A_state.elements) + len(B_state.elements)
        if total > best_total:
            best_total = total
            best_A = list(A_state.elements)
            best_B = list(B_state.elements)
        
    if best_A is not None and best_B is not None:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            polished_A, polished_B = simulated_annealing(best_A, best_B, max_time=min(remaining, 40.0))
            if len(polished_A) + len(polished_B) > best_total:
                best_A = polished_A
                best_B = polished_B
                best_total = len(best_A) + len(best_B)
                
    if best_A is None or best_B is None:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
