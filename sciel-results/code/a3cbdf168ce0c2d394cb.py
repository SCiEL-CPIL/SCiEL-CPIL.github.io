   nodes = np.sort(nodes_raw)
   nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
   for i in range(1, 8):
       if nodes[i] <= nodes[i-1]:
           nodes[i] = nodes[i-1] + 1e-6
   