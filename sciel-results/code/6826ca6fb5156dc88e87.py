class State:
    def __init__(self, points):
        self.points = points
        self.n = len(points)
        self.line_counts = {}
        self.ordinary_count = 0
        self.occupied = set(points)
        self._update_from_scratch()
        
    def _update_from_scratch(self):
        self.line_counts = {}
        self.ordinary_count = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                key = get_line_key(self.points[i], self.points[j])
                if key is None: continue
                if key not in self.line_counts:
                    self.line_counts[key] = 0
                self.line_counts[key] += 1
        
        for count in self.line_counts.values():
            if count == 2:
                self.ordinary_count += 1
                
    def move_point(self, idx, new_point):
        old_point = self.points[idx]
        if old_point == new_point:
            return 0
            
        # Update occupied set
        self.occupied.remove(old_point)
        self.occupied.add(new_point)
        
        delta_ordinary = 0
        
        for j in range(self.n):
            if j == idx:
                continue
            other_point = self.points[j]
            
            old_key = get_line_key(old_point, other_point)
            new_key = get_line_key(new_point, other_point)
            
            if old_key is None or new_key is None:
                continue
            
            old_count = self.line_counts[old_key]
            if old_count == 2:
                delta_ordinary -= 1
            old_count -= 1
            if old_count == 2:
                delta_ordinary += 1
            self.line_counts[old_key] = old_count
            
            new_count = self.line_counts.get(new_key, 0)
            if new_count == 2:
                delta_ordinary -= 1
            new_count += 1
            if new_count == 2:
                delta_ordinary += 1
            self.line_counts[new_key] = new_count
            
        self.points[idx] = new_point
        self.ordinary_count += delta_ordinary
        return delta_ordinary
