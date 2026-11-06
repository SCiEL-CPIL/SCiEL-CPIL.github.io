def _normalize_name(name):
    return name.lower().replace("_", "").replace("-", "")

# In synthesize:
n_name = _normalize_name(name)
if n_name == "cnot" and qubits == 2:
    # ...
