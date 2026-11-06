import math

def construct():
       points = []
       
       # Helper to calculate hex distance
       def hex_dist(a, b):
           return (abs(a) + abs(b) + abs(a + b)) // 2
       
       # 1. Add all points with hex_dist <= 2
       for a in range(-3, 4):
           for b in range(-3, 4):
               if hex_dist(a, b) <= 2:
                   points.append([a, b])
       
       # 2. Generate points with hex_dist == 3
       shell3 = []
       for a in range(-3, 4):
           for b in range(-3, 4):
               if hex_dist(a, b) == 3:
                   shell3.append([a, b])
       
       # 3. Sort shell3 by angle to get cyclic order
       def angle(p):
           a, b = p
           x = a + b / 2.0
           y = b * math.sqrt(3) / 2.0
           return math.atan2(y, x)
       
       shell3.sort(key=angle)
       
       # 4. Pick 13 consecutive points
       # The list is sorted cyclically. We just take the first 13.
       # Note: The "first" and "last" in the sorted list are adjacent in the cycle,
       # but since we take a contiguous block from the sorted list (which is a linearization of the cycle),
       # taking indices 0..12 gives 13 points that form a path in the cycle.
       selected_shell3 = shell3[:13]
       
       points.extend(selected_shell3)
       
       return points
