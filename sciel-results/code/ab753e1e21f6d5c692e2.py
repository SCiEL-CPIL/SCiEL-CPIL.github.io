import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(12))
    all_blocks = list(combinations(points, 6))
    all_5subs = list(combinations(points, 5))
    
    n_blocks = len(all_blocks)
    n_5subs = len(all_5subs)
    
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    class DLXNode:
        __slots__ = ['L', 'R', 'U', 'D', 'C', 'row', 'col', 'size']
        def __init__(self, col=None):
            self.L = self
            self.R = self
            self.U = self
            self.D = self
            self.C = col
            self.row = None
            self.col = col
            self.size = 0

    class DLX:
        def __init__(self, n_cols, deadline):
            self.header = DLXNode()
            self.cols = [DLXNode() for _ in range(n_cols)]
            prev = self.header
            for c in self.cols:
                c.L = prev
                c.R = self.header
                self.header.L = c
                prev = c
            for c in self.cols:
                c.U = c
                c.D = c
            self.n_cols = n_cols
            self.solution = []
            self.deadline = deadline
            
        def add_row(self, row_id, col_indices):
            if not col_indices:
                return
            first = None
            prev = None
            for col_idx in col_indices:
                node = DLXNode(self.cols[col_idx])
                node.U = self.cols[col_idx].U
                node.D = self.cols[col_idx]
                self.cols[col_idx].U.D = node
                self.cols[col_idx].U = node
                self.cols[col_idx].size += 1
                
                if first is None:
                    first = node
                    node.L = node
                    node.R = node
                else:
                    node.L = prev
                    node.R = prev.R
                    prev.R.L = node
                    prev.R = node
                prev = node
            # Set row id for all nodes in this row
            curr = first
            while True:
                curr.row = row_id
                curr = curr.R
                if curr == first:
                    break

        def cover(self, col):
            col.L.R = col.R
            col.R.L = col.L
            i = col.D
            while i != col:
                j = i.R
                while j != i:
                    j.U.D = j.D
                    j.D.U = j.U
                    j.C.size -= 1
                    j = j.R
                i = i.D

        def uncover(self, col):
            i = col.U
            while i != col:
                j = i.L
                while j != i:
                    j.C.size += 1
                    j.D.U = j
                    j.U.D = j
                    j = j.L
                i = i.U
            col.L.R = col
            col.R.L = col

        def search(self, k=0):
            if time.time() > self.deadline:
                return False
                
            if self.header.R == self.header:
                return True
                
            col = self.header.R
            min_size = float('inf')
            while col != self.header:
                if col.size < min_size:
                    min_size = col.size
                    if min_size <= 1:
                        break
                col = col.R
                
            if min_size == 0:
                return False
                
            self.cover(col)
            row = col.D
            while row != col:
                self.solution.append(row.row)
                j = row.R
                while j != row:
                    self.cover(j.C)
                    j = j.R
                if self.search(k + 1):
                    return True
                self.solution.pop()
                j = row.L
                while j != row:
                    self.uncover(j.C)
                    j = j.L
                row = row.D
            self.uncover(col)
            return False

    dlx = DLX(n_5subs, deadline)
    
    for i, b in enumerate(all_blocks):
        subs = [sub_to_idx[s] for s in combinations(b, 5)]
        dlx.add_row(i, subs)
        
    if dlx.search():
        selected_blocks = [all_blocks[i] for i in dlx.solution]
        return [list(b) for b in selected_blocks]
    else:
        # Fallback: return best effort or empty
        # Since the problem guarantees a solution, this should not happen.
        # If it does, return an empty list to avoid timeout/exception.
        return []
