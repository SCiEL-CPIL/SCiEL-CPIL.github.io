"""Rewritten program: incremental greedy for B_3[1] unique-triple-sum sets."""
import time

def construct():
    deadline = time.time() + 80.0   # safety margin under the 90s kill
    n = 4096
    selected = []
    sums = set()                     # all unordered-triple sums (with repetition) of selected

    def check_add(x):
        # sums of triples that include the new element x, against the current selected
        # unordered triple with repetition over selected + {x} that contains x:
        #   x + a + b  where a,b in selected with a <= b, and also x+x+x, x+x+a, x+a+a
        # We need every unordered triple (multiset of size 3) from selected+{x}.
        # New ones are those containing at least one x.
        # 1) x,x,x
        # 2) x,x,a  for a in selected
        # 3) x,a,b  for a <= b in selected
        s = set()
        s.add(3 * x)
        for a in selected:
            s.add(2 * x + a)
        # x + a + b for unordered pair (a,b) from selected
        m = len(selected)
        for i in range(m):
            a = selected[i]
            s.add(x + 2 * a)
            for j in range(i + 1, m):
                b = selected[j]
                s.add(x + a + b)
        # any collision with existing sums?
        if s & sums:
            return False
        sums |= s
        return True

    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if check_add(x):
            selected.append(x)
            if len(selected) >= 80:
                break
    return selected
