import importlib.util, inspect
spec = importlib.util.find_spec("autoresearch_suite.science_tcs_expansion")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
src = inspect.getsource(mod)
print("TOTAL", len(src))
print(src[30000:55000])
