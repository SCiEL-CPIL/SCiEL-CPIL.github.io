   import math
   import numpy as np
   from scipy.optimize import minimize_scalar

   APOTHEM = math.sqrt(3) / 2.0

   def construct():
       # 1. Define base cluster
       axial = [
           (0, 0),
           (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
           (1, 1), (-1, 2), (-2, 1), (2, -1)
       ]
       
       coords = []
       for q, r in axial:
           x = math.sqrt(3) * (q + r / 2.0)
           y = 1.5 * r
           coords.append((x, y))
           
       coords = np.array(coords)
       
       # 2. Center the cluster
       centroid = np.mean(coords, axis=0)
       coords = coords - centroid
       
       # 3. Scale slightly to prevent overlap errors
       scale = 1.0005
       coords = coords * scale
       
       # 4. Precompute vertices for each hexagon (relative to their center)
       # Unit hexagon vertices
       hex_verts_rel = []
       for i in range(6):
           a = i * math.pi / 3.0
           hex_verts_rel.append((math.cos(a), math.sin(a)))
       hex_verts_rel = np.array(hex_verts_rel)
       
       def get_all_vertices(theta):
           # Rotate coords
           c = np.cos(theta)
           s = np.sin(theta)
           rot = np.array([[c, -s], [s, c]])
           rotated_coords = coords @ rot.T
           
           # Rotate hex_verts_rel
           rotated_hex_verts = hex_verts_rel @ rot.T
           
           # All vertices
           # coords shape (11, 2), rotated_hex_verts shape (6, 2)
           # We need (11, 6, 2)
           all_verts = rotated_coords[:, np.newaxis, :] + rotated_hex_verts[np.newaxis, :, :]
           return all_verts.flatten() # (66, 2)
           
       def objective(theta):
           verts = get_all_vertices(theta)
           # Normals for aligned outer hexagon
           # Angles: pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
           normals = []
           for k in range(6):
               ang = math.pi / 6.0 + k * math.pi / 3.0
               normals.append((math.cos(ang), math.sin(ang)))
           normals = np.array(normals) # (6, 2)
           
           # Project all vertices onto normals
           # verts (66, 2), normals (6, 2)
           # We want max abs proj for each normal
           projs = np.abs(verts @ normals.T) # (66, 6)
           max_projs = np.max(projs, axis=0) # (6,)
           
           max_proj = np.max(max_projs)
           S = max_proj / APOTHEM
           return S
           
       # 5. Optimize theta
       res = minimize_scalar(objective, bounds=(0, math.pi/6), method='bounded')
       best_theta = res.x
       best_S = res.fun
       
       # 6. Construct result
       inner = []
       c = np.cos(best_theta)
       s = np.sin(best_theta)
       rot = np.array([[c, -s], [s, c]])
       final_coords = coords @ rot.T
       
       angle_deg = math.degrees(best_theta)
       
       for i in range(len(final_coords)):
           x, y = final_coords[i]
           inner.append([float(x), float(y), float(angle_deg)])
           
       center = [0.0, 0.0]
       side = float(best_S)
       rotation = 0.0
       
       return inner, center, side, rotation
   