def construct():
       # 3x3 grid
       points = [[x, y] for x in range(3) for y in range(3)]
       # Add 3 points "at infinity" mapped to finite plane
       # Using a projective transformation is complex, so we approximate
       # by placing them far away
       points.append([10**9, 0])
       points.append([0, 10**9])
       points.append([10**9, 10**9])
       return points
