   outer_normals = []
   for k in range(6):
       ang = phi + math.pi / 6.0 + k * math.pi / 3.0
       outer_normals.append([math.cos(ang), math.sin(ang)])
   outer_normals = np.array(outer_normals)  # (6, 2)
   