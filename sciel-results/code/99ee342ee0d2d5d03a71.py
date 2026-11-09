      def required_L(centers_subset):
          min_L = 0.0
          for (cx, cy, _, _) in centers_subset:
              for k in range(6):
                  # Pointy-Top inner hex vertices
                  angle = math.radians(60 * k + 30)
                  vx = cx + math.cos(angle)
                  vy = cy + math.sin(angle)
                  # Find minimum L such that (vx,vy) is in hex of side L
                  # Outer hex is Pointy-Top, so normals are at 0, 60, ...
                  max_proj = 0.0
                  for j in range(6):
                      ang = math.radians(60 * j)
                      proj = vx * math.cos(ang) + vy * math.sin(ang)
                      if proj > max_proj:
                          max_proj = proj
                  L_needed = max_proj / (math.sqrt(3.0) / 2.0)
                  if L_needed > min_L:
                      min_L = L_needed
          return min_L
      