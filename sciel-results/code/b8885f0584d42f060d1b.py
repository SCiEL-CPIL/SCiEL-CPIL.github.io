import time
import random
from collections import deque

N = 12

# Precompute subset indices?
# We can map frozenset to index.
# But frozenset is hashable, so dict is fine.

class AutomatonSearch:
    def __init__(self):
        self.a = [(i + 1) % N for i in range(N)]
        self.b = [max(0, i - 1) for i in range(N)]
        self.best_a = self.a[:]
        self.best_b = self.b[:]
        self.best_len = self._calc_len(self.a, self.b)
        if self.best_len == -1:
            self.a = [(i + 1) % N for i in range(N)]
            self.b = [0] * N
            self.best_a = self.a[:]
            self.best_b = self.b[:]
            self.best_len = self._calc_len(self.a, self.b)

    def _calc_len(self, a, b):
        start = frozenset(range(N))
        if len(start) == 1:
            return 0
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            dist = visited[current]
            
            # a
            next_set = frozenset(a[s] for s in current)
            if len(next_set) == 1:
                return dist + 1
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
                
            # b
            next_set = frozenset(b[s] for s in current)
            if len(next_set) == 1:
                return dist + 1
            if next_set not in visited:
                visited[next_set] = dist + 1
                queue.append(next_set)
        return -1

    def step(self):
        pos = random.randint(0, N - 1)
        letter_idx = random.randint(0, 1)
        
        new_a = self.a[:]
        new_b = self.b[:]
        
        if letter_idx == 0:
            new_val = random.randint(0, N - 1)
            new_a[pos] = new_val
        else:
            new_val = random.randint(0, N - 1)
            new_b[pos] = new_val
            
        new_len = self._calc_len(new_a, new_b)
        
        if new_len > self.best_len:
            self.best_len = new_len
            self.best_a = new_a
            self.best_b = new_b
            self.a = new_a
            self.b = new_b
        elif new_len != -1:
            if random.random() < 0.2:
                self.a = new_a
                self.b = new_b

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    searcher = AutomatonSearch()
    
    while time.time() < deadline:
        searcher.step()
        
    return [searcher.best_a, searcher.best_b]
