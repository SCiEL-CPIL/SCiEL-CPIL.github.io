if parent_key:
    p_data = dag[parent_key]
    if isinstance(p_data, dict):
        parents = p_data
    elif isinstance(p_data, list):
        # Assume list of (parent, child) or similar? 
        # Or list of lists? 
        # If list of lists, maybe p_data[i] is parents of i?
        # This is ambiguous. 
        # Let's assume it's a dict or build from edges if available.
        pass
