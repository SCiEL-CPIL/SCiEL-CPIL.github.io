        delta = 0
        affected_sets = edge_to_sets[(i,j)]
        new_energies_dict = {}
        for idx in affected_sets:
            old_e = set_energies[idx]
            adj[i][j] = new_c
            adj[j][i] = new_c
            new_e = energy_of_6set(all_6sets[idx])
            adj[i][j] = old_c
            adj[j][i] = old_c
            delta += (new_e - old_e)
            new_energies_dict[idx] = new_e
